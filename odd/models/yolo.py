from icevision.all import *
import sys

sys.path.insert(0, "./")

model_type = models.ultralytics.yolov5

class LitYOLOV5(model_type.lightning.ModelAdapter):
    def __init__(
        self,
        metrics = [COCOMetric(metric_type=COCOMetricType.bbox)],
        num_classes = 10,
        img_size = 384,
        learning_rate = 1e-4
    ):
        backbone = model_type.backbones.small
        model = model_type.model(
            backbone = backbone(pretrained=True),
            num_classes = num_classes,
            img_size = img_size
        )

        super().__init__(model, metrics)
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        return Adam(self.parameters(), lr = self.learning_rate)