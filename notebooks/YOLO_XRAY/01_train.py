# Databricks notebook source
# MAGIC %pip install pillow opencv-python sahi==0.10.8 pytorch-lightning==1.8.0.post1 icevision[all]

# COMMAND ----------

dbutils.widgets.text("data_path", "/dbfs/Users/rafael.pierre@databricks.com/security")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Predicting Forbidden Items in Baggage X-Ray Scans
# MAGIC ## Ingestion and Training
# MAGIC 
# MAGIC For the safety of the traveling public, airplane administrators operates security checkpoints at airports, seeking to keep dangerous items off airplanes. At these checkpoints, a fleet of X-ray scanners is employed, such as the Rapiscan 620DV, so Transportation Security Officers (TSOs) can inspect the contents of carry-on possessions. However, identifying and locating all potential threats can be a challenging task.
# MAGIC 
# MAGIC For this demo, we will work with a dataset collected by the TSA and Duke University. It contains a sizable new dataset of X-ray scans with a diverse set of threats in a wide array of contexts, trained several deep convolutional object detection models, and integrated such models into the Rapiscan 620DV, resulting in functional prototypes capable of operating in real time. Our objective is to train a machine learning model based on YOLOV5 - zero shot object detection model to accurately detect these items and streamline airport operations.

# COMMAND ----------

import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Getting Data From Roboflow 

# COMMAND ----------

#!pip install roboflow
#from roboflow import Roboflow

#rf = Roboflow(api_key="XXXXXXX")
#project = rf.workspace("new-workspace-fp307").project("dr-susan")
#img_dataset = project.version(2).download("coco")

# COMMAND ----------

# MAGIC %sh ls /dbfs/Users/rafael.pierre@databricks.com/security/dr-susan-2/

# COMMAND ----------

# MAGIC %sh cat /dbfs/Users/rafael.pierre@databricks.com/security/dr-susan-2/coco128.yaml

# COMMAND ----------

#parsin yml file 
# import pyyaml module
import yaml
from yaml.loader import SafeLoader

with open('/dbfs/Users/rafael.pierre@databricks.com/security/dr-susan-2/coco128.yaml', 'r') as f:
   data = list(yaml.load_all(f, Loader=SafeLoader))


# COMMAND ----------

spark.sql("CREATE DATABASE IF NOT EXISTS security")

# COMMAND ----------

annot_df = spark.createDataFrame(pd.DataFrame(data[0]['annotations']))
images_df = spark.createDataFrame(pd.DataFrame(data[0]['images']))
categories_df = spark.createDataFrame(pd.DataFrame(data[0]['categories']))

from pyspark.sql.functions import broadcast
categories_df = categories_df.withColumnRenamed("id", "category_id")
images_df = images_df.withColumnRenamed("id", "image_id")
df_images = annot_df.join(broadcast(categories_df), "category_id").join(images_df, "image_id").drop("segmentation","iscrowd","supercategory","license","date_captured")
display(df_images)

# COMMAND ----------

df_images.write.format("delta").saveAsTable("security.images_annot_categ")

# COMMAND ----------

# DBTITLE 1,First Glance
# import json
# import pandas as pd

# path = dbutils.widgets.get("data_path")
# json_path = "/dr-susan-2/train/_annotations.coco.json"
# full_path = path+json_path
# json_dict = {}

# with open(full_path, "r") as file:
#   json_dict = json.load(file)
    
# images_dict = json_dict["images"]
# annot_dict = json_dict["annotations"]

# images_df = spark.createDataFrame(pd.DataFrame.from_dict(images_dict))
# annot_df = spark.createDataFrame(pd.DataFrame.from_dict(annot_dict))
# full_set_df = images_df.join(annot_df, "id")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## to do 
# MAGIC 
# MAGIC - make a pandas udf that would feed a dict into the coco parser to speed up parsing - try to get 1 annotations convert to dict and pass it to 

# COMMAND ----------

# @pandas_udf("binary")
# def resize_image_udf(content_series):
#   def resize_image(content):
#     """resize image and serialize as jpeg"""
#     try:
#       image = Image.open(io.BytesIO(content)).resize((IMAGE_RESIZE, IMAGE_RESIZE), Image.NEAREST)
#       output = io.BytesIO()
#       image.save(output, format='JPEG')
#       return output.getvalue()
#     except Exception:
#       #some images are invalid
#       return None
  
#   new_images = content_series.apply(resize_image)
#   return new_images

# resize_image_udf

# COMMAND ----------

INPUT_DIR_IMG = "/dbfs/Users/rafael.pierre@databricks.com/security/dr-susan-2/train/"
sys.path.append(INPUT_DIR_IMG)

# COMMAND ----------

df_images = (spark.read.table("security.images_annot_categ")
             .drop("id","category_id","area")
             .withColumnRenamed("image_id", "index")
             .withColumnRenamed("name", "label")
             .withColumnRenamed("file_name", "filepath")
           
            )
display(df_images)

# COMMAND ----------

from pyspark.sql.functions import explode
display(df_images.select(explode("bbox")))

# COMMAND ----------

df_img_DF = df_images.limit(10).toPandas()
df_img_DF.head(10)

# COMMAND ----------

# index	filepath	width	height	label	xmin	ymin	xmax	ymax

# COMMAND ----------

class COTSParser(Parser):
    def __init__(self, template_record, data_dir, df):
        super().__init__(template_record=template_record)
        
        self.data_dir = data_dir
        self.df = df
        self.class_map = ClassMap(list(self.df['label'].unique()))
        
    def __iter__(self) -> Any:
        for o in self.df.itertuples():
            yield o
            
    def __len__(self) -> int:
        return len(self.df)
    
    def record_id(self, o) -> Hashable:
        return o.filepath
    
    def parse_fields(self, o, record, is_new):
        if is_new:
            record.set_filepath(os.path.join(self.data_dir,o.filepath))
            record.set_img_size(ImgSize(width=o.width, height=o.height))
            record.detection.set_class_map(self.class_map)

        #record.detection.add_bboxes([BBox.from_xyxy(o.xmin, o.ymin, o.xmax, o.ymax)])
        #<BBox (xmin:56, ymin:244, xmax:268.293579, ymax:333.875492)>
        #[203.0, 138.0, 46.247915000000006, 70.81712100000001]
        # BBoxes: [<BBox (xmin:50, ymin:78, xmax:199.321102, ymax:281.604194)>, <BBox (xmin:150, ymin:164, xmax:244.458715, ymax:307.97116600000004)>, <BBox (xmin:166, ymin:170, xmax:263.321101, ymax:319.93446900000004)>, <BBox (xmin:56, ymin:244, xmax:268.293579, ymax:333.875492)>]
        print(o.bbox)
        record.detection.add_bboxes([BBox.from_xyxy(o.bbox[2], o.bbox[3], o.bbox[0], o.bbox[1])])
        record.detection.add_labels([o.label])

# COMMAND ----------

BBox.from_xywh()

# COMMAND ----------

parser = COTSParser(template_record, INPUT_DIR_IMG, df_img_DF)

train_records, valid_records = parser.parse()
print(parser.class_map)

# COMMAND ----------

train_records.get_by_record_id(1)

# COMMAND ----------



# COMMAND ----------

import icevision
from icevision.all import *

#Class template for creating custom parser in Icevision
template_record = ObjectDetectionRecord()
Parser.generate_template(template_record)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

from icevision.parsers import COCOBBoxParser

train_parser = COCOBBoxParser(
    annotations_filepath = f"{path}/dr-susan-2/train/_annotations.coco.json",
    img_dir = f"{path}/dr-susan-2/train", idmap=)
train_parser.class_map.get_classes()

# COMMAND ----------

type(train_parser)

# COMMAND ----------

train_parser.annotations_dict

# COMMAND ----------

dir(train_parser)

# COMMAND ----------



# COMMAND ----------


CLASSES = ['background',
 'Baton',
 'Bullet',
 'Gun',
 'Hammer',
 'HandCuffs',
 'Knife',
 'Lighter',
 'Pliers',
 'Powerbank',
 'Scissors',
 'Sprayer',
 'Wrench']
class_map = ClassMap(CLASSES)
len(class_map) 

# COMMAND ----------

# DBTITLE 1,Parsing our COCO Dataset
from icevision.parsers import COCOBBoxParser
from icevision.data.data_splitter import SingleSplitSplitter

train_parser = COCOBBoxParser(
    annotations_filepath = f"{path}/dr-susan-2/train/_annotations.coco.json",
    img_dir = f"{path}/dr-susan-2/train")

valid_parser = COCOBBoxParser(
    annotations_filepath = f"{path}/dr-susan-2/valid/_annotations.coco.json",
    img_dir = f"{path}/dr-susan-2/valid")

splitter = SingleSplitSplitter()
train_records = train_parser.parse(data_splitter=splitter)
valid_records = valid_parser.parse(data_splitter=splitter)

# COMMAND ----------

train_records[0].get_by_record_id(record_id=1)

# COMMAND ----------

train_records[0].create_record_fn()

# COMMAND ----------

images_dict[0]

# COMMAND ----------

annot_dict[0]

# COMMAND ----------

import torch

 
#Check GPU availability
if not torch.cuda.is_available(): # is gpu
  raise Exception("Please use a GPU-cluster for model training, CPU instances will be too slow")


# COMMAND ----------

# DBTITLE 1,Quick Look at Some Samples
from icevision import tfms
from icevision.data import Dataset

# Transforms
# size is set to 384 because EfficientDet requires its inputs to be divisible by 128
image_size = 384
train_tfms = tfms.A.Adapter([
  *tfms.A.aug_tfms(size=image_size, presize=512),
  tfms.A.HorizontalFlip(p=0.5),
  tfms.A.RandomBrightnessContrast(p=0.2),
  tfms.A.Normalize()])

valid_tfms = tfms.A.Adapter([
  *tfms.A.resize_and_pad(image_size),
  tfms.A.Normalize()
])

# Datasets
train_ds = Dataset(train_records[0], train_tfms)
valid_ds = Dataset(valid_records[0], valid_tfms)

# COMMAND ----------

train_ds[1]

# COMMAND ----------

from icevision.visualize.show_data import show_samples

#samples = [train_ds[123] for _ in range(9)]
samples = [train_ds[i] for i in range(9)]
show_samples(samples, ncols=3)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Finetuning YOLO V5

# COMMAND ----------

from icevision import models

extra_args = {}
model_type = models.ultralytics.yolov5
backbone = model_type.backbones.small
# The yolov5 model requires an img_size parameter
extra_args['img_size'] = image_size

# Instantiate the model
model = model_type.model(
  backbone=backbone(pretrained=True),
  num_classes=len(train_parser.class_map), **extra_args) 

# COMMAND ----------

# Data Loaders
train_dl = model_type.train_dl(train_ds, batch_size=32, num_workers=4, shuffle=True)
valid_dl = model_type.valid_dl(valid_ds, batch_size=32, num_workers=4, shuffle=False)

# COMMAND ----------

# DBTITLE 1,Sample Batch
# show batch
model_type.show_batch(next(iter(valid_dl))[:10], ncols=3)

# COMMAND ----------

# DBTITLE 1,Declaring our PyTorch Lightning Model
from icevision.metrics.coco_metric.coco_metric import COCOMetric
from icevision import models
from torch.optim import Adam
import torch

metrics = [COCOMetric()]

class LightModel(models.ultralytics.yolov5.lightning.ModelAdapter):
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)

model = model.to(torch.device("cuda"))
light_model = LightModel(model, metrics=metrics).to(torch.device("cuda"))

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Training our Model
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import mlflow

early_stop_callback = EarlyStopping(
  monitor="val_loss",
  min_delta=0.00001,
  patience=5,
  verbose=True,
  mode="min"
)

experiment_name = "/Shared/security"

mlf_logger = pl_loggers.MLFlowLogger(
  experiment_name = "/Shared/security",
  save_dir = "/tmp/logs"
)

trainer = pl.Trainer(
  accelerator = "gpu",
  devices = 1,
  logger=mlf_logger,
  default_root_dir = "/tmp/yolo",
  callbacks=[early_stop_callback]
)

mlflow.set_experiment(experiment_name = experiment_name)
with mlflow.start_run(run_name = f"yolov5_{date_parsed}") as run:
  trainer.fit(light_model, train_dl, valid_dl)
  mlflow.pytorch.log_model(
    pytorch_model = light_model,
    artifact_path = "model",
    registered_model_name = "yolov5_security"
  )

# COMMAND ----------

# DBTITLE 1,Registering a Model Into Model Registry
conda_env = {
  "name": "mlflow-env",
  "channels": ["conda-forge", "anaconda"],
  "dependencies": [
    "python=3.8.10",
    "pip=22.3.1",
    "gcc_linux-64",
    "gxx_linux-64",
    "opencv",
    "pillow",
    {
      "pip": [
        "torch==1.10.2",
        "icevision[all]",
        "pytorch-lightning==1.8.0.post1",
        "sahi==0.10.8"
      ]
    }
  ]
}

with mlflow.start_run(run_name = f"yolov5_{date_parsed}") as run:
  mlflow.pytorch.log_model(
    pytorch_model = light_model.to(torch.device("cpu")),
    artifact_path = "model",
    registered_model_name = "yolov5_security",
    conda_env = conda_env
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Training model in Distributed fashion

# COMMAND ----------



# COMMAND ----------

from notebooks.spark_pytorch_distributor import MirrorRunner

from torch.optim import Adam
import torch

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

#from icevision import models
from icevision import tfms
from icevision.data import Dataset
from icevision.parsers import COCOBBoxParser
from icevision.data.data_splitter import SingleSplitSplitter
from icevision.metrics.coco_metric.coco_metric import COCOMetric
import icevision.models.ultralytics.yolov5 as Yolo5Light


import mlflow


class LightModel(Yolo5Light.lightning.ModelAdapter):
  """
  LightModel will inherit from icevision.ModelAdapter - that inherigths from LightningModelAdapter and he inherits from pl.LightningModule 
  """
  
  def configure_optimizers(self):
      return Adam(self.parameters(), lr=1e-4)


# shoiuld be included into the Class above 
def build_model(model_type, num_classes, extra_args):
  # Instantiate the model
  model = model_type.model(backbone=model_type.backbones.small(pretrained=True),
                           num_classes=num_classes, 
                           **extra_args) 
  return model
      

def parse_data(path, splitter = SingleSplitSplitter()):
  train_parser = COCOBBoxParser(
      annotations_filepath = f"{path}/dr-susan-2/train/_annotations.coco.json",
      img_dir = f"{path}/dr-susan-2/train")
  
  valid_parser = COCOBBoxParser(
      annotations_filepath = f"{path}/dr-susan-2/valid/_annotations.coco.json",
      img_dir = f"{path}/dr-susan-2/valid")

  train_records = train_parser.parse(data_splitter=splitter)
  valid_records = valid_parser.parse(data_splitter=splitter)
  
  num_classes = len(train_parser.class_map.get_classes())
  
  return (train_records, valid_records, num_classes)
  
  
def data_trfm(image_size = 384):
  # size is set to 384 because EfficientDet requires its inputs to be divisible by 128

  train_tfms = tfms.A.Adapter([
    *tfms.A.aug_tfms(size=image_size, presize=512),
    tfms.A.HorizontalFlip(p=0.5),
    tfms.A.RandomBrightnessContrast(p=0.2),
    tfms.A.Normalize()])

  valid_tfms = tfms.A.Adapter([
    *tfms.A.resize_and_pad(image_size),
    tfms.A.Normalize()])
  
  return (train_tfms, valid_tfms)


# COMMAND ----------

  
def pl_test(strat:str='dp', path:str='/dbfs/Users/rafael.pierre@databricks.com/security'):
  """
  
  This is a standard PyTorch Lighting Training loop that leverages the ddp 
  
  """

  #*****
  # Settings / Parameters 
  #*****
  BATCH_SIZE = 32
  IMAGE_SIZE = 384
  
  early_stop_callback = EarlyStopping(
                                      monitor="val_loss",
                                      min_delta=0.00001,
                                      patience=5,
                                      verbose=True,
                                      mode="min")
  
  log_path ="/dbfs/Users/anastasia.prokaieva@databricks.com/pl_training_logger"

  experiment_name = "/Shared/security"

  mlf_logger = pl_loggers.MLFlowLogger(
    experiment_name = "/Shared/security",
    save_dir = log_path
  )

  #*****
  # Data Prep / Transformation
  #******
  
  train_parser, valid_parser, num_classes = parse_data(path)
  train_tfms, valid_tfms = data_trfm(image_size=IMAGE_SIZE)

  # Datasets
  train_ds = Dataset(train_parser[0], train_tfms)
  valid_ds = Dataset(valid_parser[0], valid_tfms)
  
  #******
  # MODEL 
  #******
  
  # The yolov5 model requires an img_size parameter
  extra_args = {}
  extra_args['img_size'] = IMAGE_SIZE
  
  model_pretrained = build_model(Yolo5Light, num_classes, extra_args)
  # Data Loaders
  train_dloader = Yolo5Light.train_dl(train_ds, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
  valid_dloader = Yolo5Light.valid_dl(valid_ds, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
  
  
  metrics = [COCOMetric()]
  model = model_pretrained.to(torch.device("cuda"))
  light_model = LightModel(model, metrics=metrics).to(torch.device("cuda"))
 
  # Define the Trainer 
  trainer = pl.Trainer(
                      accelerator = "gpu", devices = 1, num_nodes=2,
                      logger=mlf_logger, strategy=strat,
                      default_root_dir = log_path,
                      max_epochs=14,
                      callbacks=[early_stop_callback]
                    )
  trainer.fit(light_model, train_dloader, valid_dloader)
  
  return trainer


# COMMAND ----------

trainer = MirrorRunner(num_slots=2, use_custom_strategy=True, local_mode=False).run(pl_test, strat='ddp') # num_slots =1, strat="dp" / num_slots=3, strat='dp', local_mode=True

# COMMAND ----------

conda_env = {
  "name": "mlflow-env",
  "channels": ["conda-forge", "anaconda"],
  "dependencies": [
    "python=3.8.10",
    "pip=22.3.1",
    "gcc_linux-64",
    "gxx_linux-64",
    "opencv",
    "pillow",
    {
      "pip": [
        "torch==1.10.2",
        "icevision[all]",
        "pytorch-lightning==1.8.0.post1",
        "sahi==0.10.8"
      ]
    }
  ]
}
date_parsed = "14_11_2022"
with mlflow.start_run(run_name = f"yolov5_{date_parsed}") as run:
  mlflow.pytorch.log_model(
    pytorch_model = trainer.model.to(torch.device("cpu")),
    artifact_path = "yolo_model_ddp",
    registered_model_name = "yolov5_security",
    conda_env = conda_env
  )

# COMMAND ----------


