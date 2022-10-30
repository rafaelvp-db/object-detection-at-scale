from odd.data.fridge import FridgeDataModule


def test_instantiate():

    fridge_data = FridgeDataModule()
    assert fridge_data is not None