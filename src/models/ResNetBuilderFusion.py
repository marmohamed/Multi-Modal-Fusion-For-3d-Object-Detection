from utils.constants import *

from models.ResnetImageFusion import *
from models.ResnetLidarBEVFusion import *
from models.ResnetLidarFV import *


class ResNetBuilderFusion(object):

    def build(self, branch, img_height, img_width, img_channels, **kwargs):
        self.img_size_1 = img_height
        self.img_size_2 = img_width
        self.c_dim = img_channels

        CONST = Const()
        if branch == CONST.IMAGE_BRANCH:
            return ResNetImageFusion(img_height, img_width, img_channels, **kwargs)
        elif branch == CONST.BEV_BRANCH:
            return ResNetLidarBEVFusion(img_height, img_width, img_channels, **kwargs)
        elif branch == CONST.FV_BRANCH:
            return ResNetLidarFV(img_height, img_width, img_channels, **kwargs)
        else:
            return None

