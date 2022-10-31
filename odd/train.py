from odd.data.fridge import FridgeDataModule
from odd.models.yolo import LitObjectDetectionModel
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import mlflow
import torch
import horovod

def train(learning_rate, max_epochs):

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
      devices=1,
      logger=mlf_logger,
      default_root_dir = "/tmp/yolo"
    )

    with mlflow.start_run(run_name = "icevision_ddp") as run:
      trainer.fit(model = model, datamodule = data)

    return model


def train_hvd(
  max_epochs = 1,
  learning_rate = 1e-4,
  batch_size = 10
):

  from torch.utils.data.distributed import DistributedSampler
  import horovod.torch as hvd
  
  try: 
    hvd.init()
    hvd_model = LitObjectDetectionModel(
      distributed = True, 
      earning_rate=learning_rate*hvd.size()
    )
    print(f"HVD Object: {hvd}")
    hvd_datamodule = FridgeDataModule(device_id = hvd.rank(), device_count = hvd.size())
    
    # Configure the sampler so that each worker gets a distinct sample of the input dataset
    train_sampler = DistributedSampler(hvd_datamodule.train_ds, num_replicas=hvd.size(), rank=hvd.rank())
    # Use train_sampler to load a different sample of data on each worker
    train_loader = torch.utils.data.DataLoader(hvd_datamodule.train_ds, batch_size=batch_size, sampler=train_sampler)
    
    # Broadcast initial parameters so all workers start with the same parameters
    hvd.broadcast_parameters(hvd_model.model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(hvd_model.optimizer, root_rank=0)
  
    for epoch in range(1, max_epochs + 1):
      # Save checkpoints only on worker 0 to prevent conflicts between workers
      print(f"Epoch: {epoch}")
      if hvd.rank() == 0:
        # we only need this on node 0
        mlflow.pytorch.autolog()

      trainer = pl.Trainer(
          max_epochs = max_epochs,
          accelerator = "gpu" if torch.cuda.is_available() else "cpu",
          devices = 1,
          default_root_dir = "/tmp/yolo"
      )

      trainer.fit(
        model = hvd_model,
        train_dataloaders = train_loader
      )

    return hvd_model
  except Exception as excp:
    print(f"Error: {str(excp)}")