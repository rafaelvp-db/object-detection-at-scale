# Databricks notebook source
import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

from icevision.all import *
import icedata

# COMMAND ----------

# Download the dataset
url = "https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip"
dest_dir = "/tmp/fridge"
data_dir = icedata.load_data(url, dest_dir)

# COMMAND ----------

# Parse annotations to create records
# Create the parser
parser = parsers.VOCBBoxParser(
  annotations_dir=data_dir / "odFridgeObjects/annotations",
  images_dir=data_dir / "odFridgeObjects/images"
)

train_records, valid_records = parser.parse()
parser.class_map

# COMMAND ----------

# Transforms
# size is set to 384 because EfficientDet requires its inputs to be divisible by 128
image_size = 384
train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=image_size, presize=512), tfms.A.Normalize()])
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])

# COMMAND ----------

# Datasets
train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)

# COMMAND ----------

# Show an element of the train_ds with augmentation transformations applied
samples = [train_ds[0] for _ in range(10)]
show_samples(samples, ncols=3)

# COMMAND ----------

extra_args = {}
model_type = models.ultralytics.yolov5
backbone = model_type.backbones.small
# The yolov5 model requires an img_size parameter
extra_args['img_size'] = image_size

# COMMAND ----------

# Instantiate the model
model = model_type.model(backbone=backbone(pretrained=True), num_classes=len(parser.class_map), **extra_args) 

# COMMAND ----------

# Data Loaders
train_dl = model_type.train_dl(train_ds, batch_size=32, num_workers=4, shuffle=True, pin_memory = True)
valid_dl = model_type.valid_dl(valid_ds, batch_size=32, num_workers=4, shuffle=False, pin_memory = True)

# COMMAND ----------

# show batch
model_type.show_batch(first(valid_dl)[:20], ncols=3)

# COMMAND ----------

metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

# COMMAND ----------

class LightModel(model_type.lightning.ModelAdapter):
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)
    
light_model = LightModel(model, metrics=metrics)

# COMMAND ----------

from pytorch_lightning import loggers as pl_loggers
import mlflow

mlflow.autolog()

mlf_logger = pl_loggers.MLFlowLogger(
  experiment_name = "/Shared/icevision",
  save_dir = "/tmp/logs"
)

trainer = pl.Trainer(
  max_epochs=50,
  accelerator = "gpu",
  devices=1,
  logger=mlf_logger,
  default_root_dir = "/tmp/yolo",
)

with mlflow.start_run(run_name = "icevision_ddp") as run:
  trainer.fit(light_model, train_dl, valid_dl)

# COMMAND ----------

infer_dl = model_type.infer_dl(valid_ds, batch_size=4, shuffle=False)
preds = model_type.predict_from_dl(model, infer_dl, keep_images=True)

# COMMAND ----------

show_preds(preds=preds[:4])

# COMMAND ----------


