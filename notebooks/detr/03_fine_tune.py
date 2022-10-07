# Databricks notebook source
!pip install --upgrade pip && pip install transformers timm -q

# COMMAND ----------

from transformers import DetrFeatureExtractor, DetrForObjectDetection

model_name = "facebook/detr-resnet-50"
feature_extractor = DetrFeatureExtractor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)

# COMMAND ----------


