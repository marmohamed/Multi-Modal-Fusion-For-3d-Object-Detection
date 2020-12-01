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
                x = conv(x, 64, kernel=3, stride=1, use_bias=True, scope='conv0_'+str(i))
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

       

            for i in range(residual_list[3]) :
                downsample_arg = (i == 0)
                x = resblock(x, channels=192, is_training=is_training, downsample=downsample_arg, scope='resblock3_' + str(i))

            x = upsample(x, size=(2, 2), scope='resnet_bev_upsample_2', use_deconv=True, filters=128, kernel_size=4)
            
            res_groups[-1] = conv(res_groups[-1], 128, kernel=1, stride=1, scope='conv2_' + str(i))
            res_groups[-1] = batch_norm(res_groups[-1], is_training=is_training, scope='bn2_' + str(i))
            res_groups[-1] = relu(res_groups[-1])

            x = x + res_groups[-1]
            x = conv(x, 128, kernel=3, stride=1, scope='conv22_' + str(i))
            x = batch_norm(x, is_training=is_training, scope='bn22_' + str(i))
            x = relu(x)

            res_groups.append(x)

            for i in range(residual_list[4]) :
                downsample_arg = (i == 0)
                x = resblock(x, channels=256, is_training=is_training, downsample=downsample_arg, scope='resblock4_' + str(i))

            x = upsample(x, size=(2, 2), scope='resnet_bev_upsample_3', use_deconv=True, filters=128, kernel_size=4)

            res_groups[-1] = conv(res_groups[-1], 128, kernel=1, stride=1, scope='conv3_' + str(i))
            res_groups[-1] = batch_norm(res_groups[-1], is_training=is_training, scope='bn3_' + str(i))
            res_groups[-1] = relu(res_groups[-1])

            x = x + res_groups[-1]
            x = conv(x, 128, kernel=3, stride=1, scope='conv32_' + str(i))
            x = batch_norm(x, is_training=is_training, scope='bn32_' + str(i))
            x = relu(x)

            res_groups.append(x)

            return x, res_groups


       