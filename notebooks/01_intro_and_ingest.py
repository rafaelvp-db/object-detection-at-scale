# Databricks notebook source
!pip install --upgrade pip && pip install fiftyone torchvision

# COMMAND ----------

from itertools import chain

from flash import Trainer
from flash.image import ObjectDetectionData, ObjectDetector
from flash.image.detection.output import FiftyOneDetectionLabelsOutput

import fiftyone as fo
import fiftyone.utils.random as four
import fiftyone.zoo as foz

# 1 Load your FiftyOne dataset
dataset = foz.load_zoo_dataset(
   "coco-2017",
   split="validation",
   max_samples=100,
   classes=["person"],
)

# Create splits from the dataset
splits = {"train": 0.7, "test": 0.1, "val": 0.1}
four.random_split(dataset, splits)

# Here we use views into one dataset,
# but you can also use a different dataset for each split
train_dataset = dataset.match_tags("train")
test_dataset = dataset.match_tags("test")
val_dataset = dataset.match_tags("val")
predict_dataset = train_dataset.take(5)

# Remove background class, it gets added by datamodule
dataset.default_classes.pop(0)

# 2 Create the Datamodule
datamodule = ObjectDetectionData.from_fiftyone(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    val_dataset=val_dataset,
    predict_dataset=predict_dataset,
    label_field="ground_truth",
    transform_kwargs={"image_size": 512},
    batch_size=4,
)

# 3 Build the model
model = ObjectDetector(
    head="efficientdet",
    backbone="d0",
    num_classes=datamodule.num_classes,
    image_size=512,
)

# 4 Create the trainer
trainer = Trainer(max_epochs=1, limit_train_batches=10)

# 5 Finetune the model
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 6 Save it!
trainer.save_checkpoint("/tmp/object_detection_model.pt")

# 7 Generate predictions
predictions = trainer.predict(
    model,
    datamodule=datamodule,
    output=FiftyOneDetectionLabelsOutput(labels=datamodule.labels),
)
predictions = list(chain.from_iterable(predictions))  # flatten batches

# Map filepaths to predictions
predictions = {p["filepath"]: p["predictions"] for p in predictions}

# Add predictions to FiftyOne dataset
dataset.set_values(
    "flash_predictions", predictions, key_field="filepath",
)

# 8 Analyze predictions in the App
session = fo.launch_app(predict_dataset)

# COMMAND ----------

 
