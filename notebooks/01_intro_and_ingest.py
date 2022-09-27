# Databricks notebook source
!pip install --upgrade pip && pip install wget

# COMMAND ----------

# Download some metadata files
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

spark.sql("CREATE DATABASE IF NOT EXISTS openimage")
spark.read.csv("dbfs:/tmp/oidv6-train-annotations-bbox.csv", header = True).write.saveAsTable("openimage.train_annotation")
spark.read.csv("dbfs:/tmp/validation-annotations-bbox.csv", header = True).write.saveAsTable("openimage.validation_annotation")
spark.read.csv("dbfs:/tmp/test-annotations-bbox.csv", header = True).write.saveAsTable("openimage.test_annotation")
spark.read.csv("dbfs:/tmp/class-descriptions-boxable.csv", schema = "class string, name string").write.saveAsTable("openimage.class")

# COMMAND ----------

spark.read.csv("dbfs:/tmp/test-images-with-rotation.csv", header = True).write.saveAsTable("openimage.test_image")
spark.read.csv("dbfs:/tmp/validation-images-with-rotation.csv", header = True).write.saveAsTable("openimage.validation_image")
spark.read.csv("dbfs:/tmp/train-images-boxable-with-rotation.csv", header = True).write.saveAsTable("openimage.train_image")

# COMMAND ----------

#TODO: filter the Image IDS which belong to the Car Class and get the corresponding Image URLs

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from openimage.train_image

# COMMAND ----------


