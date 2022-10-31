from icevision.all import *
import sys
from horovod import torch as hvd

sys.path.insert(0, "./")

model_type = models.ultralytics.yolov5

class LitObjectDetectionModel(model_type.lightning.ModelAdapter):
    def __init__(
        self,
        metrics = [COCOMetric(metric_type=COCOMetricType.bbox)],
        num_classes = 10,
        distributed = False,
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
        self.model = model
        self.learning_rate = learning_rate
        self.distributed = distributed

    def configure_optimizers(self):

        optimizer = Adam(self.parameters(), lr = self.learning_rate)
        if not self.distributed:
            self.optimizer = optimizer
            return optimizer

        dist_optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters = self.model.named_parameters()
        )

        self.optimizer = dist_optimizer
        return dist_optimizer