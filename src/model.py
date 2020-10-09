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
            'train_loc': 1,
            'train_dim': 1,
            'train_theta': 1,
            'train_dir': 1
        }
        for k in params:
            if k in defaults:
                defaults[k] = params[k]
        return defaults


    def __build_model(self):

        self.debug_layers = {}

        if self.graph is None:
            self.graph = tf.Graph()

        with self.graph.as_default():
                self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

                # self.cls_training = tf.placeholder(tf.bool, shape=(), name='cls_training')
                # self.reg_training = tf.placeholder(tf.bool, shape=(), name='reg_training')

                img_size_1 = 370
                img_size_2 = 1224
                c_dim = 3
                self.train_inputs_rgb = tf.placeholder(tf.float32, 
                                                    [None, img_size_1, img_size_2, c_dim], 
                                                    name='train_inputs_rgb')

                img_size_1 = 512
                img_size_2 = 448
                c_dim = 41
                self.train_inputs_lidar = tf.placeholder(tf.float32, 
                                    [None, img_size_1, img_size_2, c_dim], 
                                    name='train_inputs_lidar')

                self.y_true = tf.placeholder(tf.float32, shape=(None, 128, 112, 2, 9)) # target

                self.y_true_img = tf.placeholder(tf.float32, shape=(None, 24, 78, 2)) # target
                self.train_fusion_rgb = tf.placeholder(tf.bool, shape=())

                
                with tf.variable_scope("image_branch"):
                    self.cnn = ResNetBuilder().build(branch=self.CONST.IMAGE_BRANCH, img_height=370, img_width=1224, img_channels=3)
                    self.cnn.build_model(self.train_inputs_rgb, is_training=self.is_training)
                    
                with tf.variable_scope("lidar_branch"):
                    self.cnn_lidar = ResNetBuilder().build(branch=self.CONST.BEV_BRANCH, img_height=512, img_width=448, img_channels=32)
                    self.cnn_lidar.build_model(self.train_inputs_lidar, is_training=self.is_training)
                
                
                if self.params['fusion']:
                    self.cnn_lidar.res_groups2 = []
                    self.cnn.res_groups2 = []
                    with tf.variable_scope('fusion'):
                        kernels_lidar = [9, 5, 5]
                        strides_lidar = [5, 3, 3]
                        kernels_rgb = [7, 5, 5]
                        strides_rgb = [4, 3, 3]
                        lidar_loc = [0, 1, 3]
                        for i in range(3):
                          
                            att_lidar, att_rgb = AttentionFusionLayerFunc3(self.cnn.res_groups[i], None, None, self.cnn_lidar.res_groups[lidar_loc[i]], 'attention_fusion_'+str(i), is_training=self.is_training, kernel_lidar=kernels_lidar[i], kernel_rgb=kernels_rgb[i], stride_lidar=strides_lidar[i], stride_rgb=strides_rgb[i])
                          
                            with tf.variable_scope('cond'):
                                with tf.variable_scope('cond_img'):
                                    # self.cnn.res_groups[i] = tf.cond(self.train_fusion_rgb, lambda: att_rgb, lambda: self.cnn.res_groups[i])
                                    self.cnn.res_groups2.append(tf.cond(self.train_fusion_rgb, lambda: att_rgb, lambda: self.cnn.res_groups[i]))
                                with tf.variable_scope('cond_lidar'):
                                    # self.cnn_lidar.res_groups[i] = tf.cond(self.train_fusion_rgb, lambda: att_lidar, lambda: self.cnn_lidar.res_groups[i])
                                    self.cnn_lidar.res_groups2.append(tf.cond(self.train_fusion_rgb, lambda: att_lidar, lambda: self.cnn_lidar.res_groups[i]))
                                
                              
                                self.debug_layers['attention_output_rgb_'+str(i)] = att_rgb
                                self.debug_layers['attention_output_lidar_'+str(i)] = att_lidar
                else:
                    self.cnn_lidar.res_groups2 = self.cnn_lidar.res_groups
                    self.cnn.res_groups2 = self.cnn.res_groups


                with tf.variable_scope("image_branch"):

                    # print("self.cnn.res_groups")
                    # print(self.cnn.res_groups)
                    # print("self.cnn.res_groups2")
                    # print(self.cnn.res_groups2)
                    
                    if self.params['fusion']:
                        self.cnn.res_groups2.append(self.cnn.res_groups[3])

                    with tf.variable_scope("image_head"): 
                        with tf.variable_scope("fpn"): 
                            self.fpn_images = FPN(self.cnn.res_groups2, "fpn_rgb", is_training=self.is_training)

                        last_features_layer_image = self.fpn_images[2]

                        for i in range(self.params['res_blocks_image']):
                            last_features_layer_image = resblock(last_features_layer_image, 192, scope='fpn_res_'+str(i), is_training=self.is_training)

                        self.detection_layer = conv(last_features_layer_image, 2, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out')

                

                with tf.variable_scope("lidar_branch"):
                    with tf.variable_scope("fpn"): 
                        
                        fpn_lidar = FPN(self.cnn_lidar.res_groups2, "fpn_lidar", is_training=self.is_training)
                    
                        fpn_lidar[0] = maxpool2d(fpn_lidar[0], scope='maxpool_fpn0')

                        fpn_lidar = tf.concat(fpn_lidar[:], 3)

                        fpn_lidar1 = fpn_lidar[:]
                        fpn_lidar2 = fpn_lidar[:]

                        num_conv_blocks=2
                        for i in range(0, num_conv_blocks):
                            temp = conv(fpn_lidar1, 128, kernel=3, stride=1, padding='SAME', use_bias=True, scope='conv_post_fpn_1_'+str(i))
                            temp = batch_norm(temp, is_training=self.is_training, scope='bn_post_fpn_1_' + str(i))
                            temp = relu(temp)
                            # fpn_lidar = fpn_lidar + temp
                            fpn_lidar1 = temp
                            # fpn_lidar = dropout(fpn_lidar, rate=0.3, scope='fpn_lidar_dropout_'+str(i))
                            self.debug_layers['fpn_lidar_output_post_conv_1_'+str(i)] = fpn_lidar1

                        num_conv_blocks=2
                        for i in range(0, num_conv_blocks):
                            temp = conv(fpn_lidar2, 128, kernel=3, stride=1, padding='SAME', use_bias=True, scope='conv_post_fpn_2_'+str(i))
                            temp = batch_norm(temp, is_training=self.is_training, scope='bn_post_fpn_2_' + str(i))
                            temp = relu(temp)
                            # fpn_lidar = fpn_lidar + temp
                            fpn_lidar2 = temp
                            # fpn_lidar = dropout(fpn_lidar, rate=0.3, scope='fpn_lidar_dropout_'+str(i))
                            self.debug_layers['fpn_lidar_output_post_conv_2_'+str(i)] = fpn_lidar2

                       

                        if self.params['focal_loss']:
                            final_output_1_7 = conv(fpn_lidar1, 8, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_1')
                            final_output_2_7 = conv(fpn_lidar1, 8, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_2')
                            
                            final_output_1_8 = conv(fpn_lidar2, 1, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_1_8', focal_init=self.params['focal_init'])
                            final_output_2_8 = conv(fpn_lidar2, 1, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_2_8', focal_init=self.params['focal_init'])

                            final_output_1 = tf.concat([final_output_1_7, final_output_1_8], -1)
                            final_output_2 = tf.concat([final_output_2_7, final_output_2_8], -1)
                        else:
                            final_output_1 = conv(fpn_lidar, 9, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_1')
                            final_output_2 = conv(fpn_lidar, 9, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_2')
                    

                    final_output_1 = tf.expand_dims(final_output_1, 3)
                    final_output_2 = tf.expand_dims(final_output_2, 3)

                    self.debug_layers['final_layer'] = tf.concat([final_output_1, final_output_2], 3)

                    self.final_output = tf.concat([final_output_1, final_output_2], 3)

                    # self.anchors = tf.placeholder(tf.float32, [None, 128, 112, 2, 6])

                    # self.use_nms = tf.placeholder(tf.bool, shape=[])
                    # self.final_output = tf.cond(self.use_nms, lambda: nms(self.final_output, 0.5), lambda: self.final_output)


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
                        self.classification_loss, self.loc_reg_loss, self.dim_reg_loss,\
                                    self.theta_reg_loss, self.dir_reg_loss,\
                                    self.precision, self.recall, self.iou, self.iou_loc, self.iou_dim, self.theta_accuracy,\
                                    self.recall_pos, self.recall_neg, self.iou_loc_x, self.iou_loc_y, self.iou_loc_z = loss_calculator(
                                                            self.y_true,
                                                            self.final_output, 
                                                            cls_loss_instance, 
                                                            reg_loss_instance,
                                                            **loss_params)

                        
                

                        ## F1 SCORE
                        # loc_ratios = np.array([2.4375, 1., 9.375 ])
                        # self.iou_loc_weights = self.iou_loc_x * loc_ratios[0]/np.sum(loc_ratios) + self.iou_loc_y * loc_ratios[1]/np.sum(loc_ratios) + self.iou_loc_z * loc_ratios[2]/np.sum(loc_ratios)
                        # self.regression_loss = ((1-self.iou_loc_weights)*(1-self.iou)) + \
                        #              ((1-self.iou_dim)*(1-self.iou)) +\
                        #                100 * self.theta_reg_loss 
                        #             #    + 10*self.dir_reg_loss
                        # self.model_loss = 0
                        # if self.params['train_cls']:
                        #     self.model_loss += (self.recall_pos) + self.recall_neg
                        # if self.params['train_reg']:
                        #     self.model_loss += self.regression_loss
                        ## end f1 score 

                        
                        # self.model_loss += tf.cond(self.cls_training, lambda: self.recall_pos + self.recall_neg, lambda: tf.constant(0, dtype=tf.float32))
                        # self.model_loss += tf.cond(self.reg_training, lambda: self.regression_loss, lambda: tf.constant(0, dtype=tf.float32))

                        # WORKING - BEV

                      
                   

                        # self.regression_loss_bev = 0
                        # if self.params['train_loc'] == 1:
                        #     self.regression_loss_bev += 1000 * (1 - self.iou) * self.loc_reg_loss 
                        # if self.params['train_dim'] == 1:
                        #     self.regression_loss_bev += 50 * (1 - self.iou) * self.dim_reg_loss 
                        # if self.params['train_theta'] == 1:
                        #     self.regression_loss_bev += 1000 * self.theta_reg_loss 
                        # self.model_loss_bev = 0
                        # if self.params['train_cls']:
                        #     self.model_loss_bev +=  10 * (2 - self.recall - self.precision)  * self.classification_loss
                        # if self.params['train_reg']:
                        #     self.model_loss_bev +=  1 * self.regression_loss_bev
                        
                        
                        
                        # self.regression_loss_bev = 0
                        # if self.params['train_loc'] == 1:
                        #     self.regression_loss_bev += 1000 * (2 - self.iou - self.iou_loc)* self.loc_reg_loss 
                        # if self.params['train_dim'] == 1:
                        #     self.regression_loss_bev += 50 * (2 - self.iou - self.iou_dim) * self.dim_reg_loss 
                        # if self.params['train_theta'] == 1:
                        #     self.regression_loss_bev += 1000 * self.theta_reg_loss 
                        # self.model_loss_bev = 0

                        self.regression_loss_bev = 0
                        if self.params['train_loc'] == 1:
                            self.regression_loss_bev += 30 * self.loc_reg_loss 
                        if self.params['train_dim'] == 1:
                            self.regression_loss_bev += 20 * self.dim_reg_loss 
                        if self.params['train_theta'] == 1:
                            self.regression_loss_bev += 30 * self.theta_reg_loss 
                        self.model_loss_bev = 0
                        if self.params['train_cls']:
                            self.model_loss_bev +=  5 * self.classification_loss
                        if self.params['train_reg']:
                            self.model_loss_bev +=  1 * self.regression_loss_bev
                        # if self.params['train_dir'] == 1:
                        #     self.model_loss_bev += 0.1 * self.dir_reg_loss


                     
                        # self.regression_loss = tf.cond(self.train_fusion_rgb, lambda: self.regression_loss_fusion, lambda: self.regression_loss_bev)
                        # self.model_loss = tf.cond(self.train_fusion_rgb, lambda: self.model_loss_fusion, lambda: self.model_loss_bev)

                        self.regression_loss = self.regression_loss_bev
                        self.model_loss = self.model_loss_bev
                        

                        # for end to end

                        # self.regression_loss = 20 * self.loc_reg_loss + 15 * self.dim_reg_loss + 10 * self.theta_reg_loss + 0.1 * self.dir_reg_loss
                        # self.model_loss = 0
                        # if self.params['train_cls']:
                        #     self.model_loss += 0.3 * self.classification_loss
                        # if self.params['train_reg']:
                        #     self.model_loss += self.regression_loss
                        

                      


                self.model_loss_img = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_true_img, logits=self.detection_layer))
                head_only_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        "image_branch/image_head")
                self.opt_img = tf.train.AdamOptimizer(1e-3).minimize(self.model_loss_img, var_list=head_only_vars)
                self.img_only_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        "image_branch")
                self.img_only_vars.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            "fusion/cond/cond_img"))
                self.opt_img_all = tf.train.AdamOptimizer(1e-4).minimize(self.model_loss_img, var_list=self.img_only_vars)

                self.equality = tf.where(self.y_true_img >= 0.5, tf.equal(tf.cast(tf.sigmoid(self.detection_layer) >= 0.5, tf.float32), self.y_true_img), tf.zeros_like(self.y_true_img, dtype=tf.bool))
                self.accuracy = tf.reduce_sum(tf.cast(self.equality, tf.float32)) / tf.cast(tf.count_nonzero(self.y_true_img), tf.float32)



                     
                self.global_step = tf.Variable(0, name='global_step', trainable=False)

                self.lidar_only_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        "lidar_branch")
                self.lidar_only_vars.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            "fusion/cond/cond_lidar"))
                self.decay_rate = tf.train.exponential_decay(self.params['lr'], self.global_step, self.params['decay_steps'], 
                                                            self.params['decay_rate'], self.params['staircase'])  

                self.learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
                self.opt_lidar = tf.train.AdamOptimizer(self.learning_rate_placeholder)
                self.train_op_lidar = self.opt_lidar.minimize(self.model_loss,\
                                                                            var_list=self.lidar_only_vars,\
                                                                            global_step=self.global_step)
              

                if self.params['fusion']:
                    self.fusion_only_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            "fusion")  
                    self.fusion_only_vars.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            "image_branch/image_head/fpn"))
                    self.fusion_only_vars.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            "lidar_branch/fpn"))
                    self.train_op_fusion = tf.train.AdamOptimizer(1e-3).minimize(self.model_loss,\
                                                                                var_list=self.fusion_only_vars,\
                                                                                global_step=self.global_step)
                   
                else:
                    self.train_op_fusion = None


                self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.model_loss, global_step=self.global_step)

                self.saver = tf.train.Saver(max_to_keep=1)

                self.best_saver = tf.train.Saver(max_to_keep=1)

                self.lr_summary = tf.summary.scalar('learning_rate', tf.squeeze(self.decay_rate))
                self.model_loss_batches_summary = tf.summary.scalar('model_loss_batches', self.model_loss)
                self.cls_loss_batches_summary = tf.summary.scalar('classification_loss_batches', self.classification_loss)
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


                self.recall_pos_summary = tf.summary.scalar('recall_pos_summary', self.recall_pos)
                self.recall_neg_summary = tf.summary.scalar('recall_neg_summary', self.recall_neg)

                # self.iou_loc_x_summary = tf.summary.scalar('iou_loc_x_summary', self.iou_loc_x)
                # self.iou_loc_y_summary = tf.summary.scalar('iou_loc_y_summary', self.iou_loc_y)
                # self.iou_loc_z_summary = tf.summary.scalar('iou_loc_z_summary', self.iou_loc_z)
                # self.iou_loc_weights_summary = tf.summary.scalar('iou_loc_weights_summary', self.iou_loc_weights)

                self.merged = tf.summary.merge([self.lr_summary, self.model_loss_batches_summary, \
                                            self.cls_loss_batches_summary, self.reg_loss_batches_summary,\
                                            self.loc_reg_loss_batches_summary, self.dim_reg_loss_batches_summary,\
                                            self.theta_reg_loss_batches_summary, self.dir_reg_loss_batches_summary,\
                                            self.precision_summary, self.recall_summary,\
                                            self.iou_summary, self.iou_loc_summary, self.iou_dim_summary,\
                                            self.theta_accuracy_summary,\
                                            self.cls_weight_summary, self.loc_weight_summary, self.dim_weight_summary,self.theta_weight_summary,\
                                            self.recall_pos_summary, self.recall_neg_summary,\
                                            # self.iou_loc_x_summary, self.iou_loc_y_summary, self.iou_loc_z_summary, self.iou_loc_weights_summary
                                            ])


                # self.merged = tf.summary.merge([self.lr_summary, self.model_loss_batches_summary, \
                #                             self.cls_loss_batches_summary, self.reg_loss_batches_summary,\
                #                             self.loc_reg_loss_batches_summary, self.dim_reg_loss_batches_summary,\
                #                             self.theta_reg_loss_batches_summary, self.dir_reg_loss_batches_summary])

                
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
                




   