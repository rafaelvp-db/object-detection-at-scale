# Databricks notebook source
!pip install roboflow

from roboflow import Roboflow
import os

rf = Roboflow(api_key="XXXX")
project = rf.workspace("openglpro").project("stanford_car")

# set up environment
os.environ["DATASET_DIRECTORY"] = "/content/datasets"
dataset = project.version(10).download("yolov5")

# COMMAND ----------

#clone YOLOv5 and 
%cd /tmp
!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
%pip install -qr requirements.txt # install dependencies

# COMMAND ----------

import torch
import os
from IPython.display import Image, clear_output  # to display images

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

# COMMAND ----------

# MAGIC %cd /tmp/yolov5
# MAGIC !python train.py --img 416 --batch 16 --epochs 1 --data /content/datasets/Stanford_Car-10/data.yaml --weights yolov5s.pt --cache

# COMMAND ----------

# Start tensorboard
# Launch after you have started training
# logs save in the folder "runs"
%load_ext tensorboard
%tensorboard --logdir runs

# COMMAND ----------

!python detect.py --weights runs/train/exp/weights/best.pt --img 416 --conf 0.1 --source /content/datasets/Stanford_Car-10/test/images

# COMMAND ----------

#display inference on ALL test images

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import glob
import itertools

fig, ax = plt.subplots(3, 3, figsize = (10,10))

images = glob.glob('runs/detect/exp3/*.jpg')
for count, axis in enumerate(itertools.product(*(range(3), range(3)))):
  img = mpimg.imread(images[count])
  ax[axis[0], axis[1]].imshow(img)
  ax[axis[0], axis[1]].axis('off')

# COMMAND ----------


