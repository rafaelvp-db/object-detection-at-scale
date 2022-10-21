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
# MAGIC !python train.py --img 416 --batch 16 --epochs 1 --data {os.environ["DATASET_DIRECTORY"]}/Stanford_Car-10/data.yaml --weights yolov5s.pt --cache

# COMMAND ----------


