from odd.data.fridge import FridgeDataModule
from odd.models.yolo import LitObjectDetectionModel
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import mlflow
import torch
import horovod

def train(learning_rate, max_epochs, distributed = False):

    data = FridgeDataModule()
    model = LitObjectDetectionModel(
      learning_rate = learning_rate,
      distributed = False
    )
    mlflow.autolog()

    mlf_logger = pl_loggers.MLFlowLogger(
      experiment_name = "/Shared/icevision",
      save_dir = "/tmp/logs"
    )

    trainer = pl.Trainer(
      max_epochs=max_epochs,
      accelerator = "gpu" if torch.cuda.is_available() else "cpu",
      devices=4,
      logger=mlf_logger,
      default_root_dir = "/tmp/yolo",
      strategy="ddp"
    )

    with mlflow.start_run(run_name = "icevision_ddp") as run:
      trainer.fit(model = model, datamodule = data)

    return model