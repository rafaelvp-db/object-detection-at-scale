# Databricks notebook source
path = "/dbfs/Users/rafael.pierre@databricks.com/security/dr-susan-2/train/_annotations.coco.json"

# COMMAND ----------

!pip install yolov5

# COMMAND ----------

yaml = """
optimizer: 'adam'
lr0: 0.001  # initial learning rate (SGD=1E-2, Adam=1E-3)
momentum: 0.95  # momentum
weight_decay: 0.0  # optimizer weight decay
giou: 0.05  # giou loss gain
cls: 0.58  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 1.0  # obj loss gain (*=img_size/320 if img_size != 320)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.20  # iou training threshold
anchor_t: 4.0  # anchor-multiple threshold
fl_gamma: 0.0  # focal loss gamma (efficientDet default is gamma=1.5)
hsv_h: 0.014  # image HSV-Hue augmentation (fraction)
hsv_s: 0.68  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.36  # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.0  # image translation (+/- fraction)
scale: 0.5  # image scale (+/- gain)
shear: 0.0 # image shear (+/- deg)
"""

with open("./hyp.yaml", "w") as file:
  file.write(yaml)

# COMMAND ----------

from yolov5 import train, val, detect
import torch
# from yolov5.classify import train, val, predict
# from yolov5.segment import train, val, predict

train.run(imgsz=640, hyp = "hyp.yaml", data=path, device = "cuda:0", max_epochs = 1, batch_size = 32)
#val.run(imgsz=640, data='coco128.yaml', weights='yolov5s.pt')
#detect.run(imgsz=640)
#export.run(imgsz=640, weights='yolov5s.pt')

# COMMAND ----------

!cp {path} /dbfs/Users/rafael.pierre@databricks.com/security/dr-susan-2/coco128.yaml

# COMMAND ----------

!cp /dbfs/Users/rafael.pierre@databricks.com/security/dr-susan-2/coco128.yaml /dbfs/FileStore/coco128.yaml

# COMMAND ----------


