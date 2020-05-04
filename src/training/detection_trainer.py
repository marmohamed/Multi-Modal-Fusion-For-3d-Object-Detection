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
from training.clr import *

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
        self.base_lr = 0.001
        

    @abstractmethod
    def _set_params(self):
        pass


    def __prepare_dataset_feed_dict(self, dataset, train_fusion_rgb, train_fusion_fv_lidar, use_nms, anchors, is_training=True, batch_size=1):

        data = dataset.get_next(batch_size=batch_size)

        # for i in range(len(data)):
        #     data[i] = np.expand_dims(data[i], axis=0)
        if self.model.params['use_fv']:
            camera_tensor, lidar_tensor, label_tensor, Tr_velo_to_cam, R0_rect, P3, shift_h, shift_w = data
            d = {self.model.train_inputs_rgb: camera_tensor,
                    self.model.train_inputs_lidar: lidar_tensor,
                    self.model.y_true: label_tensor,
                    self.model.Tr_velo_to_cam: Tr_velo_to_cam,
                    self.model.R0_rect: R0_rect,
                    self.model.P3: P3,
                    self.model.shift_h: shift_h,
                    self.model.shift_w: shift_w,
                    self.model.use_nms: use_nms,
                    self.model.train_fusion_rgb: train_fusion_rgb,
                    self.model.train_fusion_fv_lidar: train_fusion_fv_lidar,
                    self.model.is_training: is_training,
                    self.model.anchors: anchors}
        else:
            camera_tensor, lidar_tensor, label_tensor, Tr_velo_to_cam, R0_rect, P3, shift_h, shift_w = data
            d = {self.model.train_inputs_rgb: camera_tensor,
                    self.model.train_inputs_lidar: lidar_tensor,
                    self.model.y_true: label_tensor,                   
                    self.model.Tr_velo_to_cam: Tr_velo_to_cam,
                    self.model.R0_rect: R0_rect,
                    self.model.P3: P3,
                    self.model.shift_h: shift_h,
                    self.model.shift_w: shift_w,
                    self.model.use_nms: use_nms,
                    self.model.train_fusion_rgb: train_fusion_rgb,
                    self.model.is_training: is_training,
                    self.model.anchors: anchors}
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
                    **kwargs):

        if self.dataset is None:
            self.dataset = DetectionDatasetLoader(self.data_base_path, num_samples, training_per, random_seed, training)

        self.eval_dataset = DetectionDatasetLoader(self.data_base_path, num_samples, training_per, random_seed, False)

        losses = []
        self.count_not_best = 0
        self.base_lr = 0.001
        self.last_loss = float('inf')
        self.anchors = prepare_anchors()
        self.anchors = np.repeat(self.anchors, batch_size, axis=0)
        with self.model.graph.as_default():
                
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True

                with tf.Session(config=config) as sess:
                    if restore:
                        self.model.saver.restore(sess, tf.train.latest_checkpoint('./training_files/tmp/'))
                    else:
                        sess.run(tf.global_variables_initializer())

                    counter = 0

                    for e in range(start_epoch, start_epoch+epochs, 1):
                        
                        self.dataset.reset_generator()
                        min_lr = self.get_lr(e-start_epoch)
                        print('Start epoch {0} with min_lr = {1}'.format(e, min_lr))
                        # if e - start_epoch < 10:
                        #     cls_weight = 10000
                        # else:
                        #     cls_weight = 1000
                            
                        try:
                            epoch_loss = []
                            epoch_cls_loss = []
                            epoch_reg_loss = []
                            epoch_dim_loss = []
                            epoch_loc_loss = []
                            epoch_theta_loss = []
                            epoch_dir_loss = []
                            
                            s1 = sess.run(self.model.lr_summary2, feed_dict={self.model.learning_rate_placeholder: min_lr })
                            self.model.train_writer.add_summary(s1, e)

                            while True:

                                feed_dict = self.__prepare_dataset_feed_dict(self.dataset, 
                                                                            self.branch_params['train_fusion_rgb'], 
                                                                            self.branch_params['train_fusion_fv_lidar'], 
                                                                            self.branch_params['use_nms'],
                                                                            self.anchors,
                                                                            batch_size=batch_size)

                                feed_dict[self.model.learning_rate_placeholder] = min_lr
                                

                                loss, _, classification_loss, regression_loss, loc_loss, dim_loss, theta_loss, dir_loss, summary = sess.run([self.model.model_loss, 
                                                                                                    self.branch_params['opt'], 
                                                                                                    self.model.classification_loss, 
                                                                                                    self.model.regression_loss, 
                                                                                                    self.model.loc_reg_loss,
                                                                                                    self.model.dim_reg_loss,
                                                                                                    self.model.theta_reg_loss,
                                                                                                    self.model.dir_reg_loss,
                                                                                                    self.model.merged], 
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
                                
                                if counter % 100 == 0:
                                    save_path = self.model.saver.save(sess, "./training_files/tmp/model.ckpt", global_step=self.model.global_step)
                           
                                    print("Model saved in path: %s" % save_path)
                        except (tf.errors.OutOfRangeError, StopIteration):
                            pass

                        finally:
                            save_path = self.model.saver.save(sess, "./training_files/tmp/model.ckpt", global_step=self.model.global_step)
                           
                            print("Model saved in path: %s" % save_path)

                            self.__save_summary(sess, epoch_loss, epoch_cls_loss, epoch_reg_loss, epoch_dim_loss, epoch_loc_loss, epoch_theta_loss, epoch_dir_loss, e, True)
                            
                            # losses.append(epoch_loss)
                            print('Epoch {0}: Loss = {1}, classification loss = {2}, regression_loss = {3}'.format(e, np.mean(np.array(epoch_loss).flatten()), np.mean(np.array(epoch_cls_loss).flatten()), np.mean(np.array(epoch_reg_loss).flatten())))

                            eval_batch_size=batch_size
                            d = {
                                'num_summary_images': kwargs['num_summary_images'],
                                'random_seed': random_seed,
                                'training_per': training_per,
                                'num_samples': num_samples,
                                'anchors': self.anchors}
                            self.eval(sess, e, eval_batch_size, **d)
                            
                            # try:
                            #     cls_weight = 10**(-1*len(str(np.mean(np.array(epoch_reg_loss).flatten())/np.mean(np.array(epoch_cls_loss).flatten())).split('.')[0]))
                            # except:
                            #     cls_weight = 1000

                    save_path = self.model.saver.save(sess, "./training_files/tmp/model.ckpt", global_step=self.model.global_step)
                 
                    print("Model saved in path: %s" % save_path)
                    
                # return losses

    def get_lr(self, epoch):
        # if self.count_not_best > 10:
        #     self.base_lr *= 10
        #     self.count_not_best = 0
        if self.count_not_best % 5 == 0 and self.count_not_best > 0:
            self.base_lr *= 0.1

        lr = max(self.base_lr, 1e-6)
        lr = min(self.base_lr, 1e-3)
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
                                                            self.branch_params['train_fusion_fv_lidar'], 
                                                            self.branch_params['use_nms'],
                                                            kwargs['anchors'],
                                                            is_training=False,
                                                            batch_size=batch_size)
                
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
            
            
        self.eval_dataset.reset_generator()
        th=0.5
        images = []
        list_files = list(map(lambda x: x.split('.')[0], os.listdir(self.data_base_path+'/data_object_image_3/training/image_3')))
        random.seed(kwargs['random_seed'])
        random.shuffle(list_files)

        if kwargs['num_samples'] is None:
            ln = int(len(list_files) * kwargs['training_per'])
            final_sample = len(list_files)
        else:
            ln = int(kwargs['num_samples'] * kwargs['training_per'])
            final_sample = kwargs['num_samples']

        list_files= list_files[ln:final_sample]
        eval_anchors = prepare_anchors()
        for i in range(1, kwargs['num_summary_images']+1, 1):
                
            feed_dict = self.__prepare_dataset_feed_dict(self.eval_dataset, 
                                                                self.branch_params['train_fusion_rgb'], 
                                                                self.branch_params['train_fusion_fv_lidar'], 
                                                                self.branch_params['use_nms'],
                                                                eval_anchors,
                                                                is_training=False)
            
            final_output = sess.run(self.model.final_output, feed_dict=feed_dict)

            current_file = list_files[i]
            converted_points = convert_prediction_into_real_values(final_output[0, :, :, :, :], th=th)
            points = get_points(converted_points, self.data_base_path + '/data_object_calib/training/calib/'+ current_file + '.txt', th=th)
            res = '\n'.join([' '.join([str(l) for l in points[i]]) for i in range(len(points))])
            _, labels, _, _, _, _ = read_label(res, self.data_base_path + '/data_object_calib/training/calib/'+ current_file + '.txt', 0, 0, get_actual_dims=True, from_file=False)

            img = np.clip(np.mean(feed_dict[self.model.train_inputs_lidar][0][:, :, 10:], 2), 0, 1)                
            plot_img_np = get_image(img, labels)
            images.append(plot_img_np)

        images = np.array(images)
        if self.branch_params['train_fusion_rgb']:
            s = sess.run(self.model.images_summary_fusion, feed_dict={self.model.images_summary_fusion_placeholder: images})
            self.model.validation_writer.add_summary(s, epoch)
        else:
            s = sess.run(self.model.images_summary, feed_dict={self.model.images_summary_placeholder: images})
            self.model.validation_writer.add_summary(s, epoch)
        

        


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
            'train_fusion_fv_lidar': False,
            'use_nms': False
        }

        
class FusionDetectionTrainer(DetectionTrainer):
    
    def _set_params(self):
        self.branch_params = {
            'opt': self.model.train_op_fusion,
            'train_fusion_rgb': True, 
            'train_fusion_fv_lidar': False,
            'use_nms': False
        }



