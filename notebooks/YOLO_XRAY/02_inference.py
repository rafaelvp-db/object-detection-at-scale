# Databricks notebook source
dbutils.widgets.text("data_path", "/dbfs/Users/rafael.pierre@databricks.com/security")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Predicting Forbidden Items in Baggage X-Ray Scans
# MAGIC ## Running Inference

# COMMAND ----------

import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)


# COMMAND ----------

# DBTITLE 1,Reading Test Data
from icevision.parsers import COCOBBoxParser
from icevision.data.data_splitter import SingleSplitSplitter

path = dbutils.widgets.get("data_path")
dir_path = "/dr-susan-2"
full_path = path+dir_path

test_parser = COCOBBoxParser(
    annotations_filepath = full_path+"/test/_annotations.coco.json",
    img_dir = full_path+"/test")

splitter = SingleSplitSplitter()
test_records = test_parser.parse(data_splitter = splitter)

# COMMAND ----------

# DBTITLE 1,Resizing and Normalization
from icevision import tfms
from icevision.data import Dataset

image_size = 384
test_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])
test_ds = Dataset(test_records[0], test_tfms)

# COMMAND ----------

# DBTITLE 1,Loading our Model From the Registry
import mlflow
from icevision import models

logged_model = 'runs:/a90018bb89674daaa256721e5397a17d/model'
light_model = mlflow.pytorch.load_model(model_uri = logged_model)

model_type = models.ultralytics.yolov5
infer_dl = model_type.infer_dl(test_ds, batch_size=32, shuffle=False)
preds = model_type.predict_from_dl(light_model.model, infer_dl, keep_images=True)

# COMMAND ----------

# DBTITLE 1,Showing Predicted Images
from icevision.visualize.show_data import show_preds

show_preds(preds=preds[:20])

# COMMAND ----------

model_uri = 'runs:/a90018bb89674daaa256721e5397a17d/model'

@pandas_udf("binary")
def detect_boat(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    #load the model from the registry
    model = mlflow.pyfunc.load_model(model_uri)
    for s in iterator:
        yield model.predict(s)
#save the function as SQL udf
spark.udf.register("detect_boat", detect_boat)
