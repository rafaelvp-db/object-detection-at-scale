from odd.models.yolo import LitObjectDetectionModel

def test_instantiate():
    
    model = LitObjectDetectionModel()
    assert model is not None
