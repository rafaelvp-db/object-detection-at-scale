# Databricks notebook source
import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
logging.getLogger("py4j").setLevel(logging.ERROR)

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

# DBTITLE 1,Create a folder to store our predictions results
from pathlib import Path

detection_path = "/dbfs/yolo/detections"
Path(detection_path).mkdir(exist_ok = True, parents = True)

# COMMAND ----------

# DBTITLE 1,Run inference
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import glob

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Let's take a first batch of 20 images to run predictions on
img_list = [
  f"/{file.path.replace(':', '')}"
  for file
  in dbutils.fs.ls("/tmp/open_image/test")
]

# Inference
index = np.random.randint(low = 0, high = len(img_list))
results = model(img_list[index : index + 20])
results.save(save_dir = detection_path)

# COMMAND ----------

# DBTITLE 1,Displaying the Results
# MAGIC %matplotlib inline
# MAGIC 
# MAGIC import glob
# MAGIC 
# MAGIC # Display
# MAGIC 
# MAGIC fig, ax = plt.subplots(20)
# MAGIC plt.figure(figsize=(50, 100), dpi=80)
# MAGIC fig = plt.figure(figsize=(20, 100))
# MAGIC columns = 3
# MAGIC rows = 7
# MAGIC 
# MAGIC output_path = "/dbfs/yolo/detections4"
# MAGIC img_path = glob.glob(f"{output_path}/*.jpg")
# MAGIC 
# MAGIC for i in range(1, columns * rows + 1):
# MAGIC   if i > len(img_path):
# MAGIC     break
# MAGIC   img = Image.open(img_path[i-1])
# MAGIC   im = np.asarray(img)
# MAGIC   fig.add_subplot(rows, columns, i)
# MAGIC   plt.imshow(im)
