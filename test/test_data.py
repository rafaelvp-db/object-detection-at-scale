from odd.data.fridge import FridgeDataModule


def test_instantiate():

    fridge_data = FridgeDataModule()
    assert fridge_data is not None
    assert fridge_data.train_ds is not None