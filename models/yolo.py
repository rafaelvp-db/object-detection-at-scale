import logging
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

from icevision.all import (
    Adam,
    backbone,
    COCOMetric,
    Dataset,
    show_samples,
    parsers,
    tfms
)

from icevision.models.ultralytics import yolov5

class LightModel(yolov5.lightning.ModelAdapter):
    def __init__(self, metrics, img_size, learning_rate = 1e-4):
        backbone = yolov5.backbones.small
        extra_args = {}
        extra_args['img_size'] = img_size
        model = yolov5.model(
            backbone=backbone(pretrained=True),
            num_classes=len(parser.class_map),
            **extra_args
        ) 
        super().__init__(model, metrics)
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        return Adam(self.parameters(), lr = self.learning_rate)