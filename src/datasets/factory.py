from .fer2013 import FER2013Dataset
from .ferplus import FERPlusDataset


class DatasetFactory:

    @staticmethod
    def create(name: str, config):
        name = name.lower()

        if name == "fer2013":
            return FER2013Dataset(config)

        if name == "ferplus":
            return FERPlusDataset(config)

        raise ValueError(
            f"Unknown dataset '{name}'. "
            "Valid options: ['fer2013', 'ferplus']"
        )
