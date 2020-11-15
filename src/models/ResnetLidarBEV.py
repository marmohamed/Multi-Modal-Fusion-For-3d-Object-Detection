from ops.ops import *
from utils.utils import *
from Fusion.AttentionFusionLayer import *
from models.ResNet import *

class ResNetLidarBEV(ResNet):


    def build_model(self, train_inptus, is_training=True, reuse=False, **kwargs):
        self.train_logits, self.res_groups = self.__network(train_inptus, is_training=is_training)

    def __network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("network_lidar", reuse=reuse):
           
            residual_list = [2, 4, 8, 12, 12]

            for i in range(residual_list[0]) :
                x = conv(x, 32, kernel=3, stride=1, use_bias=True, scope='conv0_'+str(i))
                x = batch_norm(x, is_training=is_training, scope='bn_res0_'+str(i))
                x = relu(x)
            
            res_groups = []

            for i in range(residual_list[1]) :
                downsample_arg = (i == 0)
                x = resblock(x, channels=64, is_training=is_training, downsample=downsample_arg, scope='resblock1_' + str(i))
            res_groups.append(x)


            for i in range(residual_list[2]) :
                downsample_arg = (i == 0)
                x = resblock(x, channels=128, is_training=is_training, downsample=downsample_arg, scope='resblock2_' + str(i))
            res_groups.append(x)

            x = upsample(x, scope='resnet_upsample_' + str(0), filters=128, use_deconv=True, kernel_size=3)

            for i in range(residual_list[3]) :
                downsample_arg = (i == 0)
                x = resblock(x, channels=192, is_training=is_training, downsample=downsample_arg, scope='resblock3_' + str(i))
            res_groups.append(x)

            x = upsample(x, scope='resnet_upsample_' + str(1), filters=192, use_deconv=True, kernel_size=3)

            for i in range(residual_list[4]) :
                downsample_arg = (i == 0)
                x = resblock(x, channels=256, is_training=is_training, downsample=downsample_arg, scope='resblock4_' + str(i))


            res_groups.append(x)

            return x, res_groups


       