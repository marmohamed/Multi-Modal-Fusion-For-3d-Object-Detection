import time
from ops.ops import *
from utils.utils import *
from models.ResNet import *

class ResNetLidarFV(ResNet):

    def build_model(self, train_inptus, is_training=True, reuse=False, **kwargs):
        self.train_logits, self.res_groups = self.__network(train_inptus, is_training, reuse)

    def __network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("network_fv", reuse=reuse):

            res_groups = []

            ch = 64 
            x = conv(x, 64, kernel=7, stride=2, padding='SAME', use_bias=True, scope='conv_0', reuse=False)
            x = batch_norm(x)
            x = relu(x)
            x = maxpool2d(x)


            x = resblock(x, channels=64, is_training=is_training, downsample=False, scope='resblock1_0')
            x = resblock(x, channels=64, is_training=is_training, downsample=False, scope='resblock1_1')

            res_groups.append(x)

            x = resblock(x, channels=128, is_training=is_training, downsample=True, scope='resblock2_0')
            x = resblock(x, channels=128, is_training=is_training, downsample=False, scope='resblock2_1')

            res_groups.append(x)

            x = resblock(x, channels=256, is_training=is_training, downsample=True, scope='resblock3_0')
            x = resblock(x, channels=256, is_training=is_training, downsample=False, scope='resblock3_1')

            res_groups.append(x)

            x = resblock(x, channels=512, is_training=is_training, downsample=True, scope='resblock4_0')
            x = resblock(x, channels=512, is_training=is_training, downsample=False, scope='resblock4_1')

            res_groups.append(x)

            return x, res_groups


       