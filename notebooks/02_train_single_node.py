# Databricks notebook source
import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

print(f"Training images: {len(dbutils.fs.ls('dbfs:/tmp/open_image/train'))}")
print(f"Testing images: {len(dbutils.fs.ls('dbfs:/tmp/open_image/test'))}")
print(f"Validation images: {len(dbutils.fs.ls('dbfs:/tmp/open_image/validation'))}")

# COMMAND ----------

! pip install --upgrade pip
! pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt  # install dependencies

# COMMAND ----------

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

%cd /dbfs/tmp #need to load YOLO into dbfs, since it's not allowed to write into repo location
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Images
imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images

# Inference
results = model(imgs)

# Display
results.save(save_dir = "/dbfs/FileStore/yolo")
img = Image.open("/dbfs/FileStore/yolo/zidane.jpg")
im = np.asarray(img)
plt.imshow(im)
