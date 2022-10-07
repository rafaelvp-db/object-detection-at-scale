# Databricks notebook source
!pip install --upgrade pip && pip install timm

# COMMAND ----------

from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image
import requests
from matplotlib import pyplot as plt

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
target_sizes = torch.tensor([image.size[::-1]])
results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

# Display image and boxes

def plot_results(results):
  plt.figure(figsize=(16,10))
  plt.imshow(image)
  ax = plt.gca()
  
  for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
      box = [round(i, 2) for i in box.tolist()]
      # let's only keep detections with score > 0.9
      if score > 0.9:
          print(
              f"Detected {model.config.id2label[label.item()]} with confidence "
              f"{round(score.item(), 3)} at location {box}"
          )
          ax.add_patch(plt.Rectangle((box[0], box[1]), box[2], box[3],
                                     fill=False, color="red", linewidth=3))
          text = f'{model.config.id2label[label.item()]}: {score:0.2f}'
          ax.text(
            box[0], box[1], text, fontsize=15,
            bbox=dict(facecolor='yellow', alpha=0.5)
          )

  plt.axis('off')
  plt.show()
  
plot_results(results)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Open Image Dataset

# COMMAND ----------

import numpy as np

# Let's take a first batch of 20 images to run predictions on
img_list = [
  f"/{file.path.replace(':', '')}"
  for file
  in dbutils.fs.ls("/tmp/open_image/test")
]

# Inference
image = Image.open(requests.get(url, stream=True).raw)
index = np.random.randint(low = 0, high = len(img_list))
image = Image.open(img_list[index])

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
target_sizes = torch.tensor([image.size[::-1]])
results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

plot_results(results)

# COMMAND ----------


