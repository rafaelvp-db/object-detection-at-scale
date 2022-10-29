import mlflow
import os
import horovod as hvd

from models.yolo import LightModel 

def train():


def train_hvd():
  hvd.init()
  
  # MLflow setup for the worker processes
  mlflow.set_tracking_uri("databricks")
  os.environ['DATABRICKS_HOST'] = db_host
  os.environ['DATABRICKS_TOKEN'] = db_token

  
  hvd_model = LitClassificationModel(class_count=5, learning_rate=1e-5*hvd.size(), device_id=hvd.rank(), device_count=hvd.size())
  hvd_datamodule = FlowersDataModule(train_converter, val_converter, device_id=hvd.rank(), device_count=hvd.size())
  
  # `gpus` parameter here should be 1 because the parallelism is controlled by Horovod
  return train(
    hvd_model,
    hvd_datamodule,
    gpus=1,
    strategy="horovod",
    device_id=hvd.rank(),
    device_count=hvd.size()
)