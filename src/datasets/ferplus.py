from .base import BaseDataset

class FERPlusDataset(BaseDataset):

    def __init__(self, config):
        self.config = config

    def load(self):
        raise NotImplementedError

    def prepare(self):
        raise NotImplementedError

    def split(self):
        raise NotImplementedError

    def get_train(self):
        raise NotImplementedError

    def get_val(self):
        raise NotImplementedError
