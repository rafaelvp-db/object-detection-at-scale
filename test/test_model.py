from odd.models.yolo import LitYOLOV5

def test_instantiate():
    
    model = LitYOLOV5()
    assert model is not None
