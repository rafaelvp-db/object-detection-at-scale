# Databricks notebook source
!pip install --upgrade pip && pip install pytorch-lightning

# COMMAND ----------

from detr import DETR
import pytorch_lightning as pl

# COMMAND ----------

class CarDataModule(pl.LightningDataModule):
  def __init__(
    self,
    batch_size,
    df,
    query = "select * from openimage.image_annotation",
    test_size = 0.2,
    augmentation = None,
    transform = None
  ):
    super().__init__()
    self.query = query
    self.batch_size = batch_size
    self.df = df
    self.image_ids = self.df['ImageID'].unique().tolist()
    self.test_size = test_size
    self._classes = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']
    self.num_classes = len(self._classes)
    self.augmentation = augmentation
    self.transform = transform

  def __len__(self):
    return len(self.image_ids)
  
  def setup(self, stage=None):
    # build dataset
    dataset = spark.sql(query).toPandas()
    self.train = dataset[dataset["Subset"] == "train"]
    self.test = dataset[dataset["Subset"] == "test"]
    self.val = dataset[dataset["Subset"] == "validation"]

    self.train.dataset.transform = self.augmentation
    self.val.dataset.transform = self.transform
    self.test.dataset.transform = self.transform

  def train_dataloader(self):
    return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=8)

  def val_dataloader(self):
    return DataLoader(self.val, batch_size=self.batch_size, num_workers=8)

  def test_dataloader(self):
    return DataLoader(self.test, batch_size=self.batch_size, num_workers=8)

  def __getitem__(self, idx):
    image_id = self.image_ids[idx]
    records = self.df[self.df['ImageID'] == image_id]
    path = records.reset_index().loc[0, "full_path"]
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0

    boxes = records[['XMin', 'YMin', 'XMax', 'YMax']].to_numpy()
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    labels = torch.ones((records.shape[0],), dtype=torch.int64)

    target = {}
    target['boxes'] = boxes
    target['labels'] = labels
    target['image_id'] = torch.tensor([idx])
    target['area'] = torch.as_tensor(area, dtype=torch.float32)
    target['iscrowd'] = torch.zeros((records.shape[0],), dtype=torch.int64)

    if self.transforms:
      sample = {
          'image': image,
          'bboxes': target['boxes'],
          'labels': labels
      }
      sample = self.transforms(**sample)
      image = sample['image']

      target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
      return image.clone().detach(), target, image_id
    
df = spark.sql("select * from openimage.image_annotation").toPandas()
car_datamodule = CarDataModule(df = df, batch_size = 4)
len(car_datamodule)

# COMMAND ----------

model = DETR(num_classes = car_datamodule.num_classes)
trainer = pl.Trainer(accelerator = "gpu", devices = 1, limit_train_batches=100, max_epochs=1)

# COMMAND ----------

trainer.fit(model=model)

# COMMAND ----------


