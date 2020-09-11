from abc import ABC, abstractmethod, ABCMeta


class ResNet(object):

    __metaclass__ = ABCMeta

    def __init__(self, img_height, img_width, img_channels):
        self.img_size_1 = img_height
        self.img_size_2 = img_width
        self.c_dim = img_channels


    @abstractmethod   
    def build_model(self, train_inptus, is_training=True, reuse=False, **kwargs):
       pass


