from icevision.models.ultralytics import yolov5

def test_instantiate():

    from models.yolo import LightModel
    model = model_type.model(backbone=backbone(pretrained=True), num_classes=len(parser.class_map), **extra_args) 
    model = LightModel()