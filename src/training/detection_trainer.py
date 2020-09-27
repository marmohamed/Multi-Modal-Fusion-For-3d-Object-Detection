import tensorflow as tf
import argparse

from tensorflow.python import debug as tf_debug
from abc import ABC, abstractmethod, ABCMeta

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
from data.segmentation_dataset_loader import *
from data.detection_dataset_loader import *
from training.Trainer import *
from training.clr import clr

from evaluation.evaluate import *
from utils.summary_images_utils import *

import math
import numpy.matlib as npm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import os



class DetectionTrainer(Trainer):

    __metaclass__ = ABCMeta

    def __init__(self, model, data_base_path, dataset):
        super(DetectionTrainer, self).__init__(model, data_base_path, dataset)
        self._set_params()
        self.count_not_best = 0
        self.base_lr = 0.0001
        

    @abstractmethod
    def _set_params(self):
        pass


    def __prepare_dataset_feed_dict(self, dataset, train_fusion_rgb, is_training=True, batch_size=1):

        data = dataset.get_next(batch_size=batch_size)
       
        camera_tensor, lidar_tensor, label_tensor, Tr_velo_to_cam, R0_rect, P3, shift_h, shift_w = data
        d = {self.model.train_inputs_rgb: camera_tensor,
                self.model.train_inputs_lidar: lidar_tensor,
                self.model.y_true: label_tensor,                   
                self.model.train_fusion_rgb: train_fusion_rgb,
                self.model.is_training: is_training
                }
        return d



    def train(self, restore=True, 
                    epochs=200, 
                    num_samples=None, 
                    training_per=0.5, 
                    random_seed=42, 
                    training=True, 
                    batch_size=1, 
                    save_steps=100,
                    start_epoch=0,
                    augment=True,
                    **kwargs):

        if self.dataset is None:
            self.dataset = DetectionDatasetLoader(self.data_base_path, num_samples, training_per, random_seed, training, augment)

        self.eval_dataset = DetectionDatasetLoader(self.data_base_path, num_samples, training_per, random_seed, False, False)


        losses = []
        self.count_not_best = 0
        # self.base_lr = self.branch_params['lr']
        self.last_loss = float('inf')
        # self.anchors = prepare_anchors()
        # self.anchors = np.repeat(self.anchors, batch_size, axis=0)
        with self.model.graph.as_default():
                
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True

                np.random.seed(random_seed)
                tf.set_random_seed(random_seed)

                with tf.Session(config=config) as sess:
                    if restore:
                        # if not self.branch_params['train_fusion_rgb']:
                        #     self.model.saver.restore(sess, tf.train.latest_checkpoint('./training_files/tmp/'))
                        # else:
                        self.model.saver.restore(sess, tf.train.latest_checkpoint('./training_files/tmp_best2/'))
                        # self.model.saver.restore(sess, './training_files/tmp_best2/model.ckpt-583440')
                    else:
                        sess.run(tf.global_variables_initializer())

                    counter = 0
                    # min_lr = 1e-6
                    for e in range(start_epoch, start_epoch+epochs, 1):
                        
                        self.dataset.reset_generator()
                        # min_lr = self.get_lr(e-start_epoch)

                        # print('Start epoch {0} with min_lr = {1}'.format(e, min_lr))
                        print('Start epoch {0}'.format(e))
                      
                        try:
                            epoch_loss = []
                            epoch_cls_loss = []
                            epoch_reg_loss = []
                            epoch_dim_loss = []
                            epoch_loc_loss = []
                            epoch_theta_loss = []
                            epoch_dir_loss = []
                            
                            # s1 = sess.run(self.model.lr_summary2, feed_dict={self.model.learning_rate_placeholder: min_lr })
                            # self.model.train_writer.add_summary(s1, e)
                            # min_lr = 1e-6
                            # s1 = sess.run(self.model.lr_summary2, feed_dict={self.model.learning_rate_placeholder: min_lr })
                            # self.model.train_writer.add_summary(s1, counter//100)
                            while True:

                                self.base_lr = clr.cyclic_learning_rate(counter, learning_rate=self.branch_params['learning_rate'], \
                                                                    max_lr=self.branch_params['max_lr'],\
                                                                    step_size=self.branch_params['step_size'],\
                                                                    mode='exp_range', gamma=.997)
                                min_lr = self.base_lr

                                s1 = sess.run(self.model.lr_summary2, feed_dict={self.model.learning_rate_placeholder: min_lr })
                                self.model.train_writer.add_summary(s1, counter)

                                feed_dict = self.__prepare_dataset_feed_dict(self.dataset, 
                                                                            self.branch_params['train_fusion_rgb'], 
                                                                            batch_size=batch_size)
                                feed_dict[self.model.learning_rate_placeholder] = min_lr

                                loss, _, classification_loss, regression_loss, loc_loss, dim_loss, theta_loss, dir_loss, summary, theta_accuracy = sess.run([self.model.model_loss, 
                                                                                                    self.branch_params['opt'], 
                                                                                                    self.model.classification_loss, 
                                                                                                    self.model.regression_loss, 
                                                                                                    self.model.loc_reg_loss,
                                                                                                    self.model.dim_reg_loss,
                                                                                                    self.model.theta_reg_loss,
                                                                                                    self.model.dir_reg_loss,
                                                                                                    self.model.merged,
                                                                                                    self.model.theta_accuracy], 
                                                                                                    feed_dict=feed_dict)

                                self.model.train_writer.add_summary(summary, counter)

                                epoch_loss.append(loss)
                                epoch_cls_loss.append(classification_loss)
                                epoch_reg_loss.append(regression_loss)
                                epoch_theta_loss.append(theta_loss)
                                epoch_dir_loss.append(dir_loss)
                                epoch_loc_loss.append(loc_loss)
                                epoch_dim_loss.append(dim_loss)
      

                                counter += 1

                        except (tf.errors.OutOfRangeError, StopIteration):
                            pass

                        finally:
                            # pass
                            save_path = self.model.saver.save(sess, "./training_files/tmp/model.ckpt", global_step=self.model.global_step)
                           
                            print("Model saved in path: %s" % save_path)

                            self.__save_summary(sess, epoch_loss, epoch_cls_loss, epoch_reg_loss, epoch_dim_loss, epoch_loc_loss, epoch_theta_loss, epoch_dir_loss, e, True)
                            
                            print('Epoch {0}: Loss = {1}, classification loss = {2}, regression_loss = {3}'.format(e, np.mean(np.array(epoch_loss).flatten()), np.mean(np.array(epoch_cls_loss).flatten()), np.mean(np.array(epoch_reg_loss).flatten())))

                            eval_batch_size=batch_size
                            d = {
                                'num_summary_images': kwargs['num_summary_images'],
                                'random_seed': random_seed,
                                'training_per': training_per,
                                'num_samples': num_samples,
                                }
                            self.eval(sess, e, eval_batch_size, **d)
                            
                            

                    save_path = self.model.saver.save(sess, "./training_files/tmp/model.ckpt", global_step=self.model.global_step)
                 
                    print("Model saved in path: %s" % save_path)
                    

    def get_lr(self, epoch):
     
        if self.count_not_best % 4 == 0 and self.count_not_best > 0:
            self.base_lr *= 0.5
        # if epoch == 35 or epoch == 50:
        #     self.base_lr *= 0.1

        lr = max(self.base_lr, 1.e-9)
        lr = min(self.base_lr, 1.e-3)
        # lr /= 4
        return lr

        

    def eval(self, sess, epoch, batch_size=1, **kwargs):
        self.eval_dataset.reset_generator()
        try:
            loss = []
            cls_loss = []
            reg_loss = []
            dim_loss = []
            loc_loss = []
            theta_loss = []
            dir_loss = []
            while True:
                
                feed_dict = self.__prepare_dataset_feed_dict(self.eval_dataset, 
                                                            self.branch_params['train_fusion_rgb'],
                                                            is_training=False,
                                                            batch_size=batch_size)

                # if epoch % 2 == 0:
                #     feed_dict[self.model.reg_training] = True
                #     feed_dict[self.model.cls_training] = False
                # else:
                #     feed_dict[self.model.reg_training] = False
                #     feed_dict[self.model.cls_training] = True
                
                all_loss, classification_loss, regression_loss, loc_loss_, dim_loss_, theta_loss_, dir_loss_ = sess.run([self.model.model_loss, 
                                                                        self.model.classification_loss,
                                                                        self.model.regression_loss,
                                                                        self.model.loc_reg_loss,
                                                                        self.model.dim_reg_loss,
                                                                        self.model.theta_reg_loss,
                                                                        self.model.dir_reg_loss], 
                                                                        feed_dict=feed_dict)

                loss.append(all_loss)
                cls_loss.append(classification_loss)
                reg_loss.append(regression_loss)
                dim_loss.append(dim_loss_)
                loc_loss.append(loc_loss_)
                theta_loss.append(theta_loss_)
                dir_loss.append(dir_loss_)
                
        except (tf.errors.OutOfRangeError, StopIteration):
            pass
        finally:
            self.__save_summary(sess, loss, cls_loss, reg_loss, dim_loss, loc_loss, theta_loss, dir_loss, epoch, False)

            print('Validation - Epoch {0}: Loss = {1}, classification loss = {2}, regression_loss = {3}'.format(epoch, np.mean(np.array(loss).flatten()), np.mean(np.array(cls_loss).flatten()), np.mean(np.array(reg_loss).flatten())))
            
            
        # self.eval_dataset.reset_generator()
        # th=0.5
        # images = []
        # list_files = list(map(lambda x: x.split('.')[0], os.listdir(self.data_base_path+'/data_object_image_3/training/image_3')))
        # random.seed(kwargs['random_seed'])
        # random.shuffle(list_files)

        # if kwargs['num_samples'] is None:
        #     ln = int(len(list_files) * kwargs['training_per'])
        #     final_sample = len(list_files)
        # else:
        #     ln = int(kwargs['num_samples'] * kwargs['training_per'])
        #     final_sample = kwargs['num_samples']

        # list_files= list_files[ln:final_sample]
        # # eval_anchors = prepare_anchors()
        # for i in range(1, kwargs['num_summary_images']+1, 1):
                
        #     feed_dict = self.__prepare_dataset_feed_dict(self.eval_dataset, 
        #                                                         self.branch_params['train_fusion_rgb'],
        #                                                         is_training=False)
            
        #     final_output = sess.run(self.model.final_output, feed_dict=feed_dict)

        #     current_file = list_files[i]
        #     converted_points = convert_prediction_into_real_values(final_output[0, :, :, :, :], th=th)
        #     points = get_points(converted_points, self.data_base_path + '/data_object_calib/training/calib/'+ current_file + '.txt', th=th)
        #     res = '\n'.join([' '.join([str(l) for l in points[i]]) for i in range(len(points))])
        #     _, labels, _, _, _, _ = read_label(res, self.data_base_path + '/data_object_calib/training/calib/'+ current_file + '.txt', 0, 0, get_actual_dims=True, from_file=False)

        #     img = np.clip(np.mean(feed_dict[self.model.train_inputs_lidar][0][:, :, 10:40], 2), 0, 1)                
        #     plot_img_np = get_image(img, labels)
        #     images.append(plot_img_np)

        # images = np.array(images)
        # if self.branch_params['train_fusion_rgb']:
        #     s = sess.run(self.model.images_summary_fusion, feed_dict={self.model.images_summary_fusion_placeholder: images})
        #     self.model.validation_writer.add_summary(s, epoch)
        # else:
        #     s = sess.run(self.model.images_summary, feed_dict={self.model.images_summary_placeholder: images})
        #     self.model.validation_writer.add_summary(s, epoch)
        

        


    def __save_summary(self, sess, epoch_loss, epoch_cls_loss, epoch_reg_loss, epoch_dim_loss, epoch_loc_loss, epoch_theta_loss, epoch_dir_loss, epoch, training=True, **kwargs):
        model_loss_summary, cls_loss_summary, reg_loss_summary, dim_loss_summary, loc_loss_summary, theta_loss_summary, dir_loss_summary = sess.run([self.model.model_loss_summary,\
                                                                         self.model.cls_loss_summary, self.model.reg_loss_summary,
                                                                         self.model.dim_loss_summary, self.model.loc_loss_summary,
                                                                         self.model.theta_loss_summary, self.model.dir_loss_summary], 
                                                                        {self.model.model_loss_placeholder: np.mean(np.array(epoch_loss).flatten()),
                                                                         self.model.cls_loss_placeholder: np.mean(np.array(epoch_cls_loss).flatten()),
                                                                         self.model.reg_loss_placeholder: np.mean(np.array(epoch_reg_loss).flatten()),
                                                                         self.model.dim_loss_placeholder: np.mean(np.array(epoch_dim_loss).flatten()),
                                                                         self.model.loc_loss_placeholder: np.mean(np.array(epoch_loc_loss).flatten()),
                                                                         self.model.theta_loss_placeholder: np.mean(np.array(epoch_theta_loss).flatten()),
                                                                         self.model.dir_loss_placeholder: np.mean(np.array(epoch_dir_loss).flatten())})

        if training:
            writer = self.model.train_writer
        else:
            writer = self.model.validation_writer

        writer.add_summary(model_loss_summary, epoch)
        writer.add_summary(cls_loss_summary, epoch)
        writer.add_summary(reg_loss_summary, epoch)
        writer.add_summary(dim_loss_summary, epoch)
        writer.add_summary(loc_loss_summary, epoch)
        writer.add_summary(theta_loss_summary, epoch)
        writer.add_summary(dir_loss_summary, epoch)

        if not training:
            if self.last_loss > np.mean(np.array(epoch_loss)):
                self.last_loss = np.mean(np.array(epoch_loss).flatten())
                save_path = self.model.best_saver.save(sess, "./training_files/tmp_best2/model.ckpt", global_step=self.model.global_step)
                           
                print("(Best) Model saved in path: %s" % save_path)
                self.count_not_best = 0
            else:
                self.count_not_best += 1

        

      

class BEVDetectionTrainer(DetectionTrainer):

    def _set_params(self):
        self.branch_params = {
            'opt': self.model.train_op_lidar,
            'train_fusion_rgb': False,
            'lr': 5e-4,
            'step_size': 2 * 1841,
            'learning_rate': 1e-6,
            'max_lr': 1e-4
        }

        
class FusionDetectionTrainer(DetectionTrainer):
    
    def _set_params(self):
        self.branch_params = {
            'opt': self.model.train_op_fusion,
            'train_fusion_rgb': True,
            'lr': 1e-5,
            'step_size': 2 * 3682,
            'learning_rate': 1e-6,
            'max_lr': 1e-4
        }


class EndToEndDetectionTrainer(DetectionTrainer):
    
    def _set_params(self):
        self.branch_params = {
            'opt': self.model.train_op,
            'train_fusion_rgb': True,
            'lr': 5e-4,
            'step_size': 2 * 3682,
            'learning_rate': 1e-6,
            'max_lr': 1e-4
        }



