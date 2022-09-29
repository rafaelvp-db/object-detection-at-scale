# Databricks notebook source
import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

! ls "/dbfs/tmp/open_image/validation" | wc -l

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select count(1), Subset, result from openimage.image_download
# MAGIC group by result, Subset
# MAGIC order by Subset, result

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## YOLO: You Only Look Once
# MAGIC 
# MAGIC <hr></hr>
# MAGIC 
# MAGIC * **YOLO** is short for You Only Look Once. It is a family of single-stage deep learning based object detectors. They are capable of more than real-time object detection with state-of-the-art accuracy.
# MAGIC * We will start off by downloading our requirements for YOLO's PyTorch version
# MAGIC * Next, we will download a pre-trained version of **YOLOv5** and check how well it does on some of the images from our dataset.

# COMMAND ----------

! pip install --upgrade pip
! pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt  # install dependencies

# COMMAND ----------

# Change the dir since local dir in repos is non-writeable
%cd /dbfs/tmp

# COMMAND ----------

from pathlib import Path

detection_path = "/dbfs/yolo/detections"
Path(detection_path).mkdir(exist_ok = True, parents = True)
results.save(save_dir = detection_path)

# COMMAND ----------

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import glob

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Images
imgs = [
  f"/{file.path}".replace(":", "/") 
  for file in dbutils.fs.ls("/tmp/open_image/test")[:20]
]

# Inference
results = model(imgs)

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC 
# MAGIC import glob
# MAGIC 
# MAGIC fig, ax = plt.subplots(20)
# MAGIC plt.figure(figsize=(50, 100), dpi=80)
# MAGIC 
# MAGIC # Display
# MAGIC 
# MAGIC fig = plt.figure(figsize=(20, 100))
# MAGIC columns = 1
# MAGIC rows = 20
# MAGIC 
# MAGIC output_path = "/dbfs/yolo/detection3"
# MAGIC img_path = glob.glob(f"{output_path}/*.jpg")
# MAGIC results.save(save_dir = output_path)
# MAGIC 
# MAGIC for i in range(1, columns*rows+1):
# MAGIC     img = Image.open(img_path[i-1])
# MAGIC     im = np.asarray(img)
# MAGIC     fig.add_subplot(rows, columns, i)
# MAGIC     plt.imshow(im)
