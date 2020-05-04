from utils.constants import *

from models.ResnetImage import *
from models.ResnetLidarBEV import *
from models.ResnetLidarFV import *


class ResNetBuilder(object):

    def build(self, branch, img_height, img_width, img_channels, **kwargs):
        self.img_size_1 = img_height
        self.img_size_2 = img_width
        self.c_dim = img_channels

        CONST = Const()
        if branch == CONST.IMAGE_BRANCH:
            return ResNetImage(img_height, img_width, img_channels, **kwargs)
        elif branch == CONST.BEV_BRANCH:
            return ResNetLidarBEV(img_height, img_width, img_channels, **kwargs)
        elif branch == CONST.FV_BRANCH:
            return ResNetLidarFV(img_height, img_width, img_channels, **kwargs)
        else:
            return None

