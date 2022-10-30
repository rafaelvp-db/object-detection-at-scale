from odd.data.fridge import FridgeDataModule
from odd.models.yolo import LitYOLOV5
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import mlflow
import torch

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


"""def train_hvd(learning_rate):
  
  # Initialize Horovod
  hvd.init()  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  if device.type == 'cuda':
    # Pin GPU to local rank
    torch.cuda.set_device(hvd.local_rank())
 
  train_dataset = datasets.MNIST(
    # Use different root directory for each worker to avoid conflicts
    root='data-%d'% hvd.rank(),  
    train=True, 
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
  )
 
  from torch.utils.data.distributed import DistributedSampler
  
  # Configure the sampler so that each worker gets a distinct sample of the input dataset
  train_sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
  # Use train_sampler to load a different sample of data on each worker
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
 
  model = Net().to(device)
  
  # The effective batch size in synchronous distributed training is scaled by the number of workers
  # Increase learning_rate to compensate for the increased batch size
  optimizer = optim.SGD(model.parameters(), lr=learning_rate * hvd.size(), momentum=momentum)
 
  # Wrap the local optimizer with hvd.DistributedOptimizer so that Horovod handles the distributed optimization
  optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
  
  # Broadcast initial parameters so all workers start with the same parameters
  hvd.broadcast_parameters(model.state_dict(), root_rank=0)
 
  for epoch in range(1, num_epochs + 1):
    train_one_epoch(model, device, train_loader, optimizer, epoch)
    # Save checkpoints only on worker 0 to prevent conflicts between workers
    if hvd.rank() == 0:
      save_checkpoint(hvd_log_dir, model, optimizer, epoch)"""