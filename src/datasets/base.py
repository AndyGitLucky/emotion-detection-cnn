from abc import ABC, abstractmethod

class BaseDataset(ABC):

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def load(self):
        """Load raw data from disk"""

    @abstractmethod
    def prepare(self):
        """Preprocess data and map labels"""

    @abstractmethod
    def split(self):
        """Create train/val/test splits"""

    @abstractmethod
    def get_train(self):
        """Return X_train, y_train"""

    @abstractmethod
    def get_val(self):
        """Return X_val, y_val"""

    def get_class_weights(self):
        return None
