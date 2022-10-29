import pytorch_lightning as pl


class COCODataModule(pl.DataModule):

    def __init__(self, url, dest_dir):
        # Download the dataset
        url = "https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip"
        dest_dir = "/tmp/fridge"
        data_dir = icedata.load_data(url, dest_dir)