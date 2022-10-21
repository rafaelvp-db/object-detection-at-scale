# Databricks notebook source
!pip install roboflow
from roboflow import Roboflow
import os

os.environ["DATASET_DIRECTORY"] = "/content/datasets"
rf = Roboflow(api_key="MZaM72PwPBoGjXpaXSms")
project = rf.workspace("openglpro").project("stanford_car")
dataset = project.version(10).download("coco")

# COMMAND ----------

!ls /content/datasets/Stanford_Car-10/train/*.json

# COMMAND ----------

!pip install 'lightning-flash[image]' -q && pip install typing-extensions --force-reinstall

# COMMAND ----------

!pip install typing-extensions --upgrade

# COMMAND ----------

# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import flash 
from flash.image import ObjectDetectionData, ObjectDetector

# 1. Create the DataModule

datamodule = ObjectDetectionData.from_coco(
    train_folder="/content/datasets/Stanford_Car-10/train/",
    train_ann_file="/content/datasets/Stanford_Car-10/train/_annotations.coco.json",
    val_split=0.2,
    transform_kwargs={"image_size": 512},
    batch_size=8,
)

# 2. Build the task
model = ObjectDetector(head="efficientdet", backbone="d0", num_classes=datamodule.num_classes, image_size=512)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=1)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# COMMAND ----------

# 4. Detect objects in a few images!
datamodule = ObjectDetectionData.from_files(
    predict_files=[
        "data/coco128/images/train2017/000000000625.jpg",
        "data/coco128/images/train2017/000000000626.jpg",
        "data/coco128/images/train2017/000000000629.jpg",
    ],
    transform_kwargs={"image_size": 512},
    batch_size=4,
)
predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("object_detection_model.pt")
