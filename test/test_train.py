from odd.train import train

def test_train():
    model = train(learning_rate = 0.1, max_epochs = 1)
    assert model is not None

def test_train_ddp():
    model = train(learning_rate = 0.1, max_epochs = 4, distributed = True)
    assert model is not None