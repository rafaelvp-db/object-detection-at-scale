import icedata
from icevision.all import parsers, tfms, Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional

FRIDGE_IMAGE_URL = "https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip"

class FridgeDataModule(pl.LightningDataModule):

    def __init__(
        self,
        url: str = FRIDGE_IMAGE_URL,
        dest_dir: str = "/tmp/fridge",
        img_size: int = 384,
        batch_size: int = 8
    ):
        super().__init__()
        self.url = url
        self.dest_dir = dest_dir
        self.batch_size = batch_size
        self.img_size = img_size

        self.train_tfms = tfms.A.Adapter([
            *tfms.A.aug_tfms(size=self.img_size, presize=512),
            tfms.A.Normalize()
        ])
        self.valid_tfms = tfms.A.Adapter([
            *tfms.A.resize_and_pad(self.img_size),
            tfms.A.Normalize()
        ])

    def prepare_data(self):
        # Download the dataset
        self.data_dir = icedata.load_data(self.url, self.dest_dir)

    def setup(self, stage: Optional[str] = None):
        parser = parsers.VOCBBoxParser(
            annotations_dir = self.data_dir / "annotations",
            images_dir = self.data_dir / "images"
        )
        train_records, valid_records = parser.parse()
        self.train_ds = Dataset(train_records, self.train_tfms)
        self.valid_ds = Dataset(valid_records, self.valid_tfms)
        self.test_ds = Dataset(valid_records, self.valid_tfms)
        self.sampler = DistributedSampler(self.train_ds)

    def train_dataloader(self):
        
        return DataLoader(self.train_ds, sampler = self.sampler, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, sampler = self.sampler, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)
