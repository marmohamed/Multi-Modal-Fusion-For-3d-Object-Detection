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
                x = conv(x, 32, kernel=3, stride=1, use_bias=False, scope='conv0_'+str(i))
                # x = conv(x, 64, kernel=3, stride=1, use_bias=False, scope='conv0_'+str(i), separable=True)
                x = batch_norm(x, is_training=is_training, scope='bn_res0_'+str(i))
                x = relu(x)
            
            res_groups = []

            for i in range(residual_list[1]) :
                downsample_arg = (i == 0)
                x = resblock(x, channels=64, is_training=is_training, downsample=downsample_arg, scope='resblock1_' + str(i))
            # x = dropout(x, 0.5, 'new_layer_dropout_0')

            res_groups.append(x)

            for i in range(residual_list[2]) :
                downsample_arg = (i == 0)
                x = resblock(x, channels=128, is_training=is_training, downsample=downsample_arg, scope='resblock2_' + str(i))
            # x = dropout(x, 0.5, 'new_layer_dropout_1')
            res_groups.append(x)

            for i in range(residual_list[3]) :
                downsample_arg = (i == 0)
                x = resblock(x, channels=256, is_training=is_training, downsample=downsample_arg, scope='resblock3_' + str(i))
            # x = dropout(x, 0.5, 'new_layer_dropout_2')
            res_groups.append(x)

            # for i in range(residual_list[4]) :
            #     downsample_arg = (i == 0)
            #     x = resblock(x, channels=256, is_training=is_training, downsample=downsample_arg, scope='resblock4_' + str(i))
            # x = dropout(x, 0.5, 'resblock_dropout_3')

            # if fusion:
            #     with tf.variable_scope('fusion_bev'):
            #         att = AttentionFusionLayerFunc(fusion_feats_rgb[3], fusion_feats_fv_lidar[3], train_fusion_fv_lidar, x, 'attention_fusion_3')
                    
            #         x_new = tf.concat([att, x], axis=-1)

            #         x_new = conv(x_new, 256, kernel=1, stride=1, scope='atention_fusion_3_post_conv', reuse=reuse)
            #         x_new = batch_norm(x_new)
            #         x_new = relu(x_new)

            #         x = tf.cond(train_fusion_rgb, lambda: x_new, lambda: x)

            # res_groups.append(x)

            return x, res_groups


       