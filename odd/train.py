from odd.data.fridge import FridgeDataModule
from odd.models.yolo import LitYOLOV5
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import mlflow
import torch
import horovod as hvd

def train(learning_rate, max_epochs):

    data = FridgeDataModule()
    model = LitYOLOV5()
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
        default_root_dir = "/tmp/yolo",
    )

    with mlflow.start_run(run_name = "icevision_ddp") as run:
        trainer.fit(model = model, datamodule = data)


def train_hvd(
  max_epochs = 1,
  learning_rate = 1e-4,
  batch_size = 10
):
  
  # Initialize Horovod
  cuda = torch.cuda.is_available()
  if cuda:
      torch.cuda.set_device(hvd.local_rank())
      torch.cuda.manual_seed()
  hvd.init()  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  if device.type == 'cuda':
    # Pin GPU to local rank
    torch.cuda.set_device(hvd.local_rank())
 
  from torch.utils.data.distributed import DistributedSampler
  
  data = FridgeDataModule()
  # Configure the sampler so that each worker gets a distinct sample of the input dataset
  train_sampler = DistributedSampler(data.train_ds, num_replicas=hvd.size(), rank=hvd.rank())
  # Use train_sampler to load a different sample of data on each worker
  train_loader = torch.utils.data.DataLoader(data.train_ds, batch_size=batch_size, sampler=train_sampler)
 
  module = LitYOLOV5(learning_rate = learning_rate * hvd.size())
  optimizer = module.optimizer
 
  # Wrap the local optimizer with hvd.DistributedOptimizer so that Horovod handles the distributed optimization
  optimizer = hvd.DistributedOptimizer(
    optimizer,
    named_parameters = module.model.named_parameters()
  )
  
  # Broadcast initial parameters so all workers start with the same parameters
  hvd.broadcast_parameters(module.model.state_dict(), root_rank=0)
 
  for epoch in range(1, max_epochs + 1):
    
    # Save checkpoints only on worker 0 to prevent conflicts between workers
    if hvd.rank() == 0:
      