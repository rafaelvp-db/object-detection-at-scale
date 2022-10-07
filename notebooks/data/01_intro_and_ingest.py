# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Data Ingestion
# MAGIC ## Google Open Image Datasets
# MAGIC 
# MAGIC <hr></hr>
# MAGIC 
# MAGIC * We will train a model to detect different types of automobiles in pictures.
# MAGIC * To train our model, we will leverage Google's Open Image project datasets.
# MAGIC * First step is to download CSV files for image **annotations**, as well as a list of image categories (e.g. Cat, Dog, Car, Ambulance etc).
# MAGIC * Once we download these text files, we will then proceed to download the actual images that will be used for our training. Since we are talking about hundreds of thousands of images, we will leverage the power of Spark to download these images at scale.

# COMMAND ----------

# Download some metadata/annotation files
!wget https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv -O /dbfs/tmp/class-descriptions-boxable.csv
!wget -O /dbfs/tmp/oid_bbox_trainable_label_map.pbtxt https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/oid_bbox_trainable_label_map.pbtxt
!wget https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv -O /dbfs/tmp/test-annotations-bbox.csv
!wget https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv -O /dbfs/tmp/validation-annotations-bbox.csv
!wget https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv -O /dbfs/tmp/oidv6-train-annotations-bbox.csv
  
!cat /dbfs/tmp/class-descriptions-boxable.csv | grep Car

# COMMAND ----------

!wget https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv -O /dbfs/tmp/test-images-with-rotation.csv
!wget https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv -O /dbfs/tmp/validation-images-with-rotation.csv
!wget https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv -O /dbfs/tmp/train-images-boxable-with-rotation.csv

# COMMAND ----------

from pyspark.sql import functions as F

spark.sql("DROP DATABASE IF EXISTS openimage CASCADE")
spark.sql("CREATE DATABASE IF NOT EXISTS openimage")
spark.read.csv("dbfs:/tmp/oidv6-train-annotations-bbox.csv", header = True).withColumn("split", F.lit("train")).write.saveAsTable("openimage.train_annotation", mode = "overwrite")
spark.read.csv("dbfs:/tmp/validation-annotations-bbox.csv", header = True).withColumn("split", F.lit("validation")).write.saveAsTable("openimage.validation_annotation", mode = "overwrite")
spark.read.csv("dbfs:/tmp/test-annotations-bbox.csv", header = True).withColumn("split", F.lit("test")).write.saveAsTable("openimage.test_annotation", mode = "overwrite")
spark.read.csv("dbfs:/tmp/class-descriptions-boxable.csv", schema = "class string, name string").write.saveAsTable("openimage.class")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from openimage.train_annotation

# COMMAND ----------

spark.read.csv("dbfs:/tmp/test-images-with-rotation.csv", header = True).write.saveAsTable("openimage.test_image")
spark.read.csv("dbfs:/tmp/validation-images-with-rotation.csv", header = True).write.saveAsTable("openimage.validation_image")
spark.read.csv("dbfs:/tmp/train-images-boxable-with-rotation.csv", header = True).write.saveAsTable("openimage.train_image")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from openimage.train_image

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select distinct a.ImageID, i.OriginalURL, i.Subset
# MAGIC from (
# MAGIC   select ImageID, LabelName from
# MAGIC   (
# MAGIC     select distinct ImageId, LabelName from openimage.train_annotation as train
# MAGIC     union distinct select distinct ImageID, LabelName from openimage.validation_annotation as val
# MAGIC     union distinct select distinct ImageID, LabelName from openimage.test_annotation as test
# MAGIC   )
# MAGIC ) as a
# MAGIC inner join openimage.class as c
# MAGIC on a.LabelName = c.class
# MAGIC inner join (
# MAGIC   select distinct ImageID, OriginalURL, Subset from openimage.train_image
# MAGIC   union distinct select ImageID, OriginalURL, Subset from openimage.test_image
# MAGIC   union distinct select ImageID, OriginalURL, Subset from openimage.validation_image
# MAGIC ) as i
# MAGIC on a.ImageID = i.ImageID
# MAGIC where c.name in ('Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck')

# COMMAND ----------

_sqldf.write.saveAsTable("openimage.training_image_url", mode = "overwrite")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from openimage.training_image_url

# COMMAND ----------

# DBTITLE 1,Dowloading Images
import time
from pyspark.sql.functions import udf, col
import pyspark

spark.conf.set("spark.databricks.optimizer.adaptive.enabled", "true")
spark.conf.set("spark.default.parallelism", "240")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "100485760")

def download_image(
  subset,
  url,
  image_id,
  root_path = "/dbfs/tmp/open_image"
):
  
  import wget
  import sys
  sys.stdout.fileno = lambda: 0 #workaround for wget
  from pathlib import Path
  from urllib.error import HTTPError

  folder_path = f"{root_path}/{subset}"
  file_name = f"{image_id}_{url.split('/')[-1]}"
  try:
    if not Path(f"{folder_path}/{file_name}").exists():
      wget.download(
        url,
        out = f"{folder_path}/{file_name}"
      )
    return "ok"
  except HTTPError as e:
    return "nok"

udf_download_images = udf(download_image)

result_df = _sqldf
result_df = result_df.repartition(spark.sparkContext.defaultParallelism).cache()
result_df.count()

result_df = result_df \
  .withColumn(
    "result",
    udf_download_images(col("Subset"), col("OriginalURL"), col("ImageID"))
  ).write\
  .saveAsTable(
    "openimage.image_download",
    mode = "overwrite"
  )

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select count(1) from openimage.image_download
# MAGIC where result = 'ok'

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from openimage.image_download

# COMMAND ----------

from pyspark.sql import functions as F

dbfs_path = "/dbfs/tmp/open_image"

df = spark.sql(
  """
    select t.ImageID, t.XMin, t.XMax, t.YMin, t.YMax, d.OriginalURL, d.Subset from openimage.image_download d
      inner join (
        select ImageID, XMin, XMax, YMin, YMax from openimage.train_annotation
        union all select ImageID, XMin, XMax, YMin, YMax from openimage.test_annotation
        union all select ImageID, XMin, XMax, YMin, YMax from openimage.validation_annotation
      ) as t
      on d.ImageID = t.ImageId
  """) \
  .withColumn(
    "file_name", F.split(F.col("d.OriginalURL"), '/').getItem(4)
  ) \
  .withColumn(
    "full_path", F.concat(F.lit(dbfs_path), F.col("file_name"))
  )

df.count()

# COMMAND ----------

df.write.saveAsTable("openimage.image_annotation")
