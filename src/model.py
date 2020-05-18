import tensorflow as tf
import argparse
import numpy as np
from tensorflow.python import debug as tf_debug
import os

from utils.constants import *
from utils.utils import *
from utils.anchors import *
from utils.nms import *

from loss.losses import *

from models.ResNetBuilder import *
from models.ResnetImage import *
from models.ResnetLidarBEV import *
from models.ResnetLidarFV import *

from FPN.FPN import *

from Fusion.FusionLayer import *

class Model(object):

    def __init__(self, graph=None, **params):
        self.CONST = Const()
        self.graph = graph
        self.params = self.__prepare_parameters(**params)
        self.__build_model()


    def __prepare_parameters(self, **params):
        defaults = {
            'focal_loss': True,
            'weight_loss': False,
            'focal_init': -1.99,
            'lr': 5e-4, 
            'decay_steps': 5000,
            'decay_rate': 0.9,
            'staircase': False,
            'train_cls': True,
            'train_reg': True,
            'fusion': True,
            'mse_loss': False,
            'res_blocks': 0,
            'res_blocks_image': 1,
            'use_fv': False
        }
        for k in params:
            if k in defaults:
                defaults[k] = params[k]
        return defaults


    def __build_model(self):
        if self.graph is None:
            self.graph = tf.Graph()
        # self.strategy = tf.distribute.MirroredStrategy()
        # with self.strategy.scope():
        with self.graph.as_default():
                # self.cls_weight = tf.placeholder(tf.float32, shape=(), name='cls_weight')
                self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
                self.batch_size = tf.placeholder(tf.int32, shape=(), name='batch_size')

                img_size_1 = 370
                img_size_2 = 1224
                c_dim = 3
                self.train_inputs_rgb = tf.placeholder(tf.float32, 
                                                    [None, img_size_1, img_size_2, c_dim], 
                                                    name='train_inputs_rgb')

                if self.params['use_fv']:
                    c_dim = 64
                    self.train_inputs_fv_lidar = tf.placeholder(tf.float32, 
                                                        [None, img_size_1, img_size_2, c_dim], 
                                                        name='train_inputs_fv_lidar')

                img_size_1 = 512
                img_size_2 = 448
                c_dim = 40
                self.train_inputs_lidar = tf.placeholder(tf.float32, 
                                    [None, img_size_1, img_size_2, c_dim], 
                                    name='train_inputs_lidar')
                self.label_weights = tf.placeholder(tf.float32, shape=(None, 128, 112, 2, 1)) # target

                self.y_true = tf.placeholder(tf.float32, shape=(None, 128, 112, 2, 9)) # target

                self.y_true_img = tf.placeholder(tf.float32, shape=(None, 24, 78, 2)) # target

                self.Tr_velo_to_cam = tf.placeholder(tf.float32, shape=(None, 4, 4))
                self.R0_rect = tf.placeholder(tf.float32, shape=(None, 4, 4))
                self.P3 = tf.placeholder(tf.float32, shape=(None, 3, 4))
                self.shift_h =  tf.placeholder(tf.float32, shape=[None, 1])
                self.shift_w = tf.placeholder(tf.float32, shape=[None, 1])

                self.train_fusion_rgb = tf.placeholder(tf.bool, shape=())
                if self.params['use_fv']:
                    self.train_fusion_fv_lidar = tf.placeholder(tf.bool, shape=())

                with tf.variable_scope("image_branch"):
                    self.cnn = ResNetBuilder().build(branch=self.CONST.IMAGE_BRANCH, img_height=370, img_width=1224, img_channels=3)
                    self.cnn.build_model(self.train_inputs_rgb, is_training=self.is_training)
                    
                if self.params['use_fv']:
                    with tf.variable_scope("lidar_fv_branch"):
                        self.cnn_fv_lidar = ResNetBuilder().build(branch=self.CONST.FV_BRANCH, img_height=370, img_width=1224, img_channels=32)
                        self.cnn_fv_lidar.build_model(self.train_inputs_fv_lidar, is_training=self.is_training)
                        fpn_fv_lidar = FPN(self.cnn_fv_lidar.res_groups, "fpn_fv_lidar")
                        fpn_fv_lidar[2] = conv(fpn_fv_lidar[2], 192, kernel=1, stride=1, scope='post_fv_conv', reuse=False)

                self.debug_layers = {}

                with tf.variable_scope("lidar_branch"):
                    self.cnn_lidar = ResNetBuilder().build(branch=self.CONST.BEV_BRANCH, img_height=512, img_width=448, img_channels=32)
                    self.cnn_lidar.build_model(self.train_inputs_lidar, is_training=self.is_training)
                    x_temp = self.cnn_lidar.train_logits
                    
                if self.params['fusion']:
                    with tf.variable_scope('fusion'):
                        if self.params['use_fv']:
                            att = AttentionFusionLayerFunc(last_features_layer_image, fpn_fv_lidar[2], self.train_fusion_fv_lidar, x_temp, 'attention_fusion_3', self.batch_size)
                            self.debug_layers['attention_output'] = att
                            x_new = tf.concat([att, x_temp], axis=-1)
                            with tf.variable_scope('post_fusion_conv'):
                                x_new = conv(x_new, 256, kernel=1, stride=1, scope='atention_fusion_3_post_conv', reuse=False)
                                x_new = batch_norm(x_new, is_training=self.is_training)
                                x_new = relu(x_new)
                                self.debug_layers['attention_module_output'] = x_new
                        else:
                            kernels_lidar = [9, 5, 5]
                            strides_lidar = [5, 3, 3]
                            kernels_rgb = [7, 5, 5]
                            strides_rgb = [4, 3, 3]
                            for i in range(3):
                                print('-----------------------------------')
                                print('START ', str(i))
                                att_lidar, att_rgb = AttentionFusionLayerFunc3(self.cnn.res_groups[i], None, None, self.cnn_lidar.res_groups[i], 'attention_fusion_'+str(i), kernel_lidar=kernels_lidar[i], kernel_rgb=kernels_rgb[i], stride_lidar=strides_lidar[i], stride_rgb=strides_rgb[i])
                                self.cnn.res_groups[i] = att_rgb
                                self.cnn_lidar.res_groups[i] = att_lidar
                                self.debug_layers['attention_output_rgb_'+str(i)] = att_rgb
                                self.debug_layers['attention_output_lidar_'+str(i)] = att_lidar

                with tf.variable_scope("image_branch"):
                   
                    with tf.variable_scope("image_head"): 
                        self.fpn_images = FPN(self.cnn.res_groups, "fpn_rgb", is_training=self.is_training)

                        last_features_layer_image = self.fpn_images[2]

                        for i in range(self.params['res_blocks_image']):
                            last_features_layer_image = resblock(last_features_layer_image, 192, scope='fpn_res_'+str(i))

                        self.detection_layer = conv(last_features_layer_image, 2, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out')

                self.model_loss_img = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_true_img, logits=self.detection_layer))
            
                head_only_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        "image_branch/image_head")

                self.opt_img = tf.train.AdamOptimizer(1e-3).minimize(self.model_loss_img, var_list=head_only_vars)

                img_only_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        "image_branch")

                self.opt_img_all = tf.train.AdamOptimizer(1e-4).minimize(self.model_loss_img, var_list=img_only_vars)


                self.equality = tf.where(self.y_true_img >= 0.5, tf.equal(tf.cast(tf.sigmoid(self.detection_layer) >= 0.5, tf.float32), self.y_true_img), tf.zeros_like(self.y_true_img, dtype=tf.bool))
                self.accuracy = tf.reduce_sum(tf.cast(self.equality, tf.float32)) / tf.cast(tf.count_nonzero(self.y_true_img), tf.float32)




                with tf.variable_scope("lidar_branch"):
                    # if not self.params['fusion']:
                    #     x_new = x_temp
                    # x_temp = tf.cond(self.train_fusion_rgb, lambda: x_new, lambda: x_temp)

                    # self.cnn_lidar.res_groups.append(x_temp)
                    fpn_lidar = FPN(self.cnn_lidar.res_groups[:3], "fpn_lidar", is_training=self.is_training)
                    # fpn_lidar = FPN(self.cnn_lidar.res_groups[:], "fpn_lidar", is_training=self.is_training)

                    self.debug_layers['fpn_lidar'] = fpn_lidar
                    
                    fpn_lidar[0] = maxpool2d(fpn_lidar[0], scope='maxpool_fpn0')
                    fpn_lidar[2] = upsample(fpn_lidar[2], size=(2, 2), scope='fpn_upsample_1', use_deconv=True, kernel_size=4)
                    # fpn_lidar[3] = upsample(fpn_lidar[3], size=(4, 4), scope='fpn_upsample_2', use_deconv=True)

                    fpn_lidar = tf.concat(fpn_lidar[:], 3)
                 
                    self.debug_layers['fpn_lidar_output'] = fpn_lidar

                    num_conv_blocks=1
                    for i in range(num_conv_blocks):
                        fpn_lidar = conv(fpn_lidar, 128, kernel=3, stride=1, padding='SAME', use_bias=True, scope='conv_post_fpn_'+str(i))
                        fpn_lidar = batch_norm(fpn_lidar, scope='bn_post_fpn_' + str(i))
                        fpn_lidar = relu(fpn_lidar)
                        self.debug_layers['fpn_lidar_output_post_conv_'+str(i)] = fpn_lidar

                    if self.params['focal_loss']:
                        final_output_1_7 = conv(fpn_lidar, 8, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_1')
                        final_output_2_7 = conv(fpn_lidar, 8, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_2')
                        
                        # final_output_1_8 = conv(fpn_lidar, 1, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_1_8', focal_init=self.params['focal_init'])
                        # final_output_2_8 = conv(fpn_lidar, 1, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_2_8', focal_init=self.params['focal_init'])
                    
                        final_output_1_8 = conv(fpn_lidar, 1, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_1_8')
                        final_output_2_8 = conv(fpn_lidar, 1, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_2_8')
                    

                        final_output_1 = tf.concat([final_output_1_7, final_output_1_8], -1)
                        final_output_2 = tf.concat([final_output_2_7, final_output_2_8], -1)
                    else:
                        final_output_1 = conv(fpn_lidar, 9, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_1')
                        final_output_2 = conv(fpn_lidar, 9, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_2')
                    

                    final_output_1 = tf.expand_dims(final_output_1, 3)
                    final_output_2 = tf.expand_dims(final_output_2, 3)

                    self.debug_layers['final_layer'] = tf.concat([final_output_1, final_output_2], 3)

                    self.final_output = tf.concat([final_output_1, final_output_2], 3)

                    self.anchors = tf.placeholder(tf.float32, [None, 128, 112, 2, 6])

                    self.use_nms = tf.placeholder(tf.bool, shape=[])
                    self.final_output = tf.cond(self.use_nms, lambda: nms(self.final_output, 0.5), lambda: self.final_output)

                    # self.final_output = adjust_predictions(self.final_output, self.anchors)

                    ############################
                    #  under lidar_branch scope
                    ############################
                    with tf.variable_scope("loss_weights"):
                        self.loc_weight = tf.get_variable('loc_weight', shape=(), initializer=tf.constant_initializer(1), dtype=tf.float32)
                        self.dim_weight = tf.get_variable('dim_weight', shape=(), initializer=tf.constant_initializer(1), dtype=tf.float32)
                        self.theta_weight = tf.get_variable('theta_weight', shape=(), initializer=tf.constant_initializer(1), dtype=tf.float32)
                        self.cls_weight = tf.get_variable('cls_weight', shape=(), initializer=tf.constant_initializer(1), dtype=tf.float32)

                with tf.variable_scope('Loss'):
                        cls_loss_instance = ClsLoss('classification_loss')
                        reg_loss_instance = RegLoss('regression_loss')
                        loss_calculator = LossCalculator()
                        loss_params = {'focal_loss': self.params['focal_loss'], 'weight': self.params['weight_loss'], 'mse': self.params['mse_loss']}
                        self.classification_loss, self.precision, self.recall,\
                                    self.loc_reg_loss, self.dim_reg_loss,\
                                    self.iou, self.iou_loc, self.iou_dim,\
                                    self.theta_reg_loss, self.theta_accuracy,\
                                    self.dir_reg_loss = loss_calculator(
                                                            self.y_true,
                                                            self.final_output, 
                                                            cls_loss_instance, 
                                                            reg_loss_instance,
                                                            **loss_params)

                        
                        

                        # self.regression_loss = ((1-self.iou_loc)*(1-self.iou)) * tf.exp(-self.loc_weight) * self.loc_reg_loss + self.loc_weight +\
                        #              ((1-self.iou_dim)*(1-self.iou)) * tf.exp(-self.dim_weight) * self.dim_reg_loss + self.dim_weight +\
                        #               ((1-self.theta_accuracy)**2) * tf.exp(-self.theta_weight) * self.theta_reg_loss + self.theta_weight
                        #             #    + 10*self.dir_reg_loss
                        # self.model_loss = 0
                        # if self.params['train_cls']:
                        #     self.model_loss += (1-self.precision)*(1-self.recall) * tf.exp(-self.cls_weight) * self.classification_loss + self.cls_weight
                        # if self.params['train_reg']:
                        #     self.model_loss += self.regression_loss

                        self.regression_loss = 100 * self.loc_reg_loss + 50 * self.dim_reg_loss + 500 * self.theta_reg_loss + 10*self.dir_reg_loss
                        self.model_loss = 0
                        if self.params['train_cls']:
                            self.model_loss += self.classification_loss
                        if self.params['train_reg']:
                            self.model_loss += self.regression_loss

                        # self.regression_loss = (1-self.iou_loc)*(1-self.iou) + (1-self.iou_dim)*(1-self.iou) + 500*self.theta_weight
                        # self.model_loss = 0
                        # if self.params['train_cls']:
                        #     # self.model_loss +=  -101*((self.precision * self.recall)/ (self.precision + 100 * self.recall + 1e-8))
                        #     self.model_loss +=  4 + (-self.precision - self.recall)
                        # if self.params['train_reg']:
                        #     self.model_loss += self.regression_loss
                     
                self.global_step = tf.Variable(0, name='global_step', trainable=False)

                self.lidar_only_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        "lidar_branch")
                self.decay_rate = tf.train.exponential_decay(self.params['lr'], self.global_step, self.params['decay_steps'], 
                                                            self.params['decay_rate'], self.params['staircase'])  

                self.learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
                self.opt_lidar = tf.train.AdamOptimizer(self.learning_rate_placeholder)
                self.train_op_lidar = self.opt_lidar.minimize(self.model_loss,\
                                                                            var_list=self.lidar_only_vars,\
                                                                            global_step=self.global_step)
              

                if self.params['fusion']:
                    fusion_only_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            "fusion")  
                    # fusion_only_vars.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                    #                         "image_branch/fpn_rgb"))
                    self.train_op_fusion = tf.train.AdamOptimizer(1e-3).minimize(self.model_loss,\
                                                                                var_list=fusion_only_vars,\
                                                                                global_step=self.global_step)
                    if self.params['use_fv']:
                        fv_only_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                "lidar_fv_branch") 
                        fv_only_vars.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                "fv_fusion"))
                        fv_only_vars.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                "fusion/post_fusion_conv"))
                        self.train_op_fv = tf.train.AdamOptimizer(1e-3).minimize(self.model_loss,\
                                                                                    var_list=fv_only_vars,\
                                                                                    global_step=self.global_step)

                else:
                    self.train_op_fusion = None


                self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.model_loss, global_step=self.global_step)


                self.saver = tf.train.Saver(max_to_keep=2)

                self.best_saver = tf.train.Saver(max_to_keep=2)

                self.lr_summary = tf.summary.scalar('learning_rate', tf.squeeze(self.decay_rate))
                self.model_loss_batches_summary = tf.summary.scalar('model_loss_batches', self.model_loss)
                self.cls_loss_batches_summary = tf.summary.scalar('classification_loss_batches', self.classification_loss)
                # self.cls_loss_2_batches_summary = tf.summary.scalar('classification_loss_2_batches', self.classification_loss_2)
                self.reg_loss_batches_summary = tf.summary.scalar('regression_loss_batches', self.regression_loss)
                self.loc_reg_loss_batches_summary = tf.summary.scalar('loc_regression_loss_batches', self.loc_reg_loss)
                self.dim_reg_loss_batches_summary = tf.summary.scalar('dim_regression_loss_batches', self.dim_reg_loss)
                self.theta_reg_loss_batches_summary = tf.summary.scalar('theta_regression_loss_batches', self.theta_reg_loss)
                self.dir_reg_loss_batches_summary = tf.summary.scalar('dir_regression_loss_batches', self.dir_reg_loss)

                self.precision_summary = tf.summary.scalar('precision_batches', self.precision)
                self.recall_summary = tf.summary.scalar('recall_batches', self.recall)

                self.iou_summary = tf.summary.scalar('iou_batches', self.iou)
                self.iou_loc_summary = tf.summary.scalar('iou_loc_batches', self.iou_loc)
                self.iou_dim_summary = tf.summary.scalar('iou_dim_batches', self.iou_dim)
                self.theta_accuracy_summary = tf.summary.scalar('theta_accuracy_batches', self.theta_accuracy)

                self.cls_weight_summary = tf.summary.scalar('cls_weight_summary', self.cls_weight)
                self.loc_weight_summary = tf.summary.scalar('loc_weight_summary', self.loc_weight)
                self.dim_weight_summary = tf.summary.scalar('dim_weight_summary', self.dim_weight)
                self.theta_weight_summary = tf.summary.scalar('theta_weight_summary', self.theta_weight)

                self.merged = tf.summary.merge([self.lr_summary, self.model_loss_batches_summary, \
                                            self.cls_loss_batches_summary, self.reg_loss_batches_summary,\
                                            self.loc_reg_loss_batches_summary, self.dim_reg_loss_batches_summary,\
                                            self.theta_reg_loss_batches_summary, self.dir_reg_loss_batches_summary,\
                                            self.precision_summary, self.recall_summary,\
                                            self.iou_summary, self.iou_loc_summary, self.iou_dim_summary,\
                                            self.theta_accuracy_summary,\
                                            self.cls_weight_summary, self.loc_weight_summary, self.dim_weight_summary,self.theta_weight_summary])

                
                self.model_loss_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
                self.model_loss_summary = tf.summary.scalar('model_loss', self.model_loss_placeholder)
                self.cls_loss_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
                self.cls_loss_summary = tf.summary.scalar('classification_loss', self.cls_loss_placeholder)
                self.reg_loss_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
                self.reg_loss_summary = tf.summary.scalar('regression_loss', self.reg_loss_placeholder)

                self.theta_loss_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
                self.theta_loss_summary = tf.summary.scalar('theta_loss', self.theta_loss_placeholder)
                self.dir_loss_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
                self.dir_loss_summary = tf.summary.scalar('dir_loss', self.dir_loss_placeholder)
                self.loc_loss_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
                self.loc_loss_summary = tf.summary.scalar('loc_loss', self.loc_loss_placeholder)
                self.dim_loss_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
                self.dim_loss_summary = tf.summary.scalar('dim_loss', self.dim_loss_placeholder)

                self.lr_summary2 = tf.summary.scalar('lr_ph', self.learning_rate_placeholder)

                



                self.images_summary_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
                self.images_summary = tf.summary.image('images', self.images_summary_placeholder)

                self.images_summary_fusion_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
                self.images_summary_fusion = tf.summary.image('images_fusion', self.images_summary_fusion_placeholder)

                self.images_summary_segmentation_cars_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 24, 78, 1])
                self.images_summary_segmentation_cars = tf.summary.image('images_segmantation_cars', self.images_summary_segmentation_cars_placeholder)
                self.images_summary_segmentation_road_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 24, 78, 1])
                self.images_summary_segmentation_road = tf.summary.image('images_segmentation_road', self.images_summary_segmentation_road_placeholder)

                self.accuracy_image_summary_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
                self.accuracy_image_summary = tf.summary.scalar('accuracy_image', self.accuracy_image_summary_placeholder)
                self.model_loss_image_summary_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
                self.model_loss_image_summary = tf.summary.scalar('model_loss_image', self.model_loss_image_summary_placeholder)

                self.train_writer = tf.summary.FileWriter('./training_files/train', self.graph)
                self.validation_writer = tf.summary.FileWriter('./training_files/test')
                




   