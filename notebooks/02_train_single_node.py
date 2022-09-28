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

# COMMAND ----------

dbutils.fs.ls("/tmp/open_image/train")[0]

# COMMAND ----------

# DBTITLE 1,Testing with Car Images
# MAGIC %cd /dbfs/tmp
# MAGIC 
# MAGIC from PIL import Image
# MAGIC import numpy as np
# MAGIC import matplotlib.pyplot as plt
# MAGIC import torch
# MAGIC 
# MAGIC 
# MAGIC model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# MAGIC 
# MAGIC # Images
# MAGIC imgs = ["/dbfs/tmp/open_image/train/0000048549557964_18233009494_029b52ca79_o.jpg"]
# MAGIC 
# MAGIC # Inference
# MAGIC results = model(imgs)
# MAGIC 
# MAGIC # Display
# MAGIC results.save(save_dir = "/dbfs/FileStore/yolo_car")

# COMMAND ----------

dbutils.fs.ls("/tmp/open_image/train/")[100]

# COMMAND ----------

img_path = "/dbfs/tmp/open_image/train/001156eb13f37194_6409333165_ecceed029b_o.jpg"

img = Image.open(img_path)
im = np.asarray(img)
plt.imshow(im)

# COMMAND ----------


