# Databricks notebook source
import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

print(f"Training images: {len(dbutils.fs.ls('dbfs:/tmp/open_image/train'))}")
print(f"Testing images: {len(dbutils.fs.ls('dbfs:/tmp/open_image/test'))}")
print(f"Validation images: {len(dbutils.fs.ls('dbfs:/tmp/open_image/validation'))}")

# COMMAND ----------

!pip install --upgrade pip && pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt  # install dependencies

# COMMAND ----------

# MAGIC %cd /dbfs/tmp
# MAGIC 
# MAGIC # Model
# MAGIC model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# COMMAND ----------

from matplotlib import pyplot as plt
import matplotlib.image as mpimg

# Images
imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images

# Inference
results = model(imgs)

# Results
#results.print()
results.show()
#results.xyxy[0]  # img1 predictions (tensor)
#results.pandas().xyxy[0]  # img1 predictions (pandas)

# COMMAND ----------

results.save(save_dir = "/dbfs/FileStore/yolo.jpg")

displayHTML('<img src="/FileStore/yolo.jpg"/>')

# COMMAND ----------

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("/tmp/yolo/test.jpg/zidane.jpg")
im = np.asarray(img)
plt.imshow(im)

# COMMAND ----------


