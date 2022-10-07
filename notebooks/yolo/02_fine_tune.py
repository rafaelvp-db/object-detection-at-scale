# Databricks notebook source
!pip install --upgrade pip && pip install pytorch-lightning

# COMMAND ----------

from pytorch_lightning import LightningModule, Trainer
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

# COMMAND ----------

class StanfordCarsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size


        # Augmentation policy for training set
        self.augmentation = transforms.Compose([
              transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
              transforms.RandomRotation(degrees=15),
              transforms.RandomHorizontalFlip(),
              transforms.CenterCrop(size=224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose([
              transforms.Resize(size=256),
              transforms.CenterCrop(size=224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        
        self.num_classes = 196


    def prepare_data(self):
        pass


    def setup(self, stage=None):
        # build dataset
        dataset = StanfordCars(root=self.data_dir, download=True, split="train")
        # split dataset
        self.train, self.val = random_split(dataset, [6500, 1644])


        self.test = StanfordCars(root=self.data_dir, download=True, split="test")
        
        self.test = random_split(self.test, [len(self.test)])[0]


        self.train.dataset.transform = self.augmentation
        self.val.dataset.transform = self.transform
        self.test.dataset.transform = self.transform
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=8)


    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=8)


    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=8)


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
    self.image_ids = self.df['ImageID'].unique().tolist()
    self.transforms = transforms
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

# COMMAND ----------

df = spark.sql("select * from openimage.image_annotation").toPandas()
car_dataset = CarDataset(df = df)
len(car_dataset)

# COMMAND ----------

import torchvision.models as models

class YoloTransferLearning(LightningModule):
    def __init__(self):
        super().__init__()
        self._classes = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']
        backbone = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = len(self._classes)
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        ...

# COMMAND ----------

model = ImagenetTransferLearning()
trainer = Trainer()
trainer.fit(model)
