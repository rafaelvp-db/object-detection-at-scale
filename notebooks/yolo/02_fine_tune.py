# Databricks notebook source
!pip install --upgrade pip && pip install pytorch-lightning

# COMMAND ----------

from pytorch_lightning import LightningModule, Trainer
import pytorch_lightning as pl

# COMMAND ----------

class CarDataset(object):
    def __init__(self, df, IMG_DIR, transforms=None):
        self.a = 0
        self.df = df
        self.img_dir = IMG_DIR
        self.image_ids = self.df['img_path'].unique().tolist()
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_ids)
        
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        records = self.df[self.df['img_path'] == image_id]
        image = cv2.imread(self.img_dir+image_id,cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].to_numpy()
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
