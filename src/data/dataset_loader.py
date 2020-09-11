from abc import ABC, abstractmethod, ABCMeta


class DatasetLoader(object):

    __metaclass__ = ABCMeta

    def __init__(self, base_path, num_samples=None, training_per=0.5, random_seed=0, training=True, augment=False, **kwargs):
        self.base_path = base_path
        self.num_samples = num_samples
        self.training_per = training_per
        self.random_seed = random_seed
        self.training = training
        self.augment=augment

        self.params = self._defaults(**kwargs)
        self.generator = self._init_generator()

    @abstractmethod
    def _defaults(self, **kwargs):
        pass

    @abstractmethod
    def _init_generator(self):
        pass

    @abstractmethod
    def reset_generator(self):
        pass

    @abstractmethod
    def get_next(self, batch_size=1):
        pass


