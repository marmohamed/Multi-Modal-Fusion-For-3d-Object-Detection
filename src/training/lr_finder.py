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

from evaluation.evaluate import *
from utils.summary_images_utils import *

import math
import numpy.matlib as npm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import json




class LRFinder:

    def __init__(self, model, trainer, data_base_path, dataset):
        self.model = model
        self.trainer = trainer
        self.data_base_path = data_base_path
        self.dataset = dataset

        

    def find(self, epochs=3, lower_bound=1e-5, upper_bound=1e-2, step_size=10, loss_batch_size=32, **kwargs):

        with self.model.graph.as_default():
                
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True

                with tf.Session(config=config) as sess:
                    

                    self.model.lr_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
                    opt = tf.train.AdamOptimizer(self.model.lr_placeholder).minimize(self.model.model_loss,\
                                                                            var_list=self.model.lidar_only_vars,\
                                                                            global_step=self.model.global_step)
                    sess.run(tf.global_variables_initializer())

                    j = 0
                    current_lr = lower_bound
                                        
                    stats = dict()
                    e = 0
                    stop_training = False
                    j = 0
                    try:
                        while e < epochs and not stop_training:
                            self.dataset.reset_generator()
                            print('Start epoch {0}'.format(e))
                            try:

                                while not stop_training:
                                    j += 1
                                    epoch_loss, epoch_cls_loss, epoch_reg_loss = self.trainer.train_for_batches(sess,
                                                                                                                    opt,
                                                                                                                    self.dataset,
                                                                                                                    batches=loss_batch_size,
                                                                                                                    lr=current_lr)
                                    print('finish batch ', j)
                                    if epoch_loss is not None:
                                        stats[current_lr] = {
                                                'model_loss_mean': str(np.mean(np.array(epoch_loss))),
                                                'cls_loss_mean': str(np.mean(np.array(epoch_cls_loss))),
                                                'reg_loss_mean': str(np.mean(np.array(epoch_reg_loss)))
                                            }

                                        if j % 100 == 0:
                                            json_stats = json.dumps(stats)
                                            f = open("./training_files/losses.json","w")
                                            f.write(json_stats)
                                            f.close()
                                        
                                        current_lr = current_lr * step_size
                                        if current_lr > upper_bound:
                                            stop_training = True

                                    else:
                                        raise StopIteration()
                                    
                            except (tf.errors.OutOfRangeError, StopIteration):
                                pass

                            finally:
                                save_path = self.model.saver.save(sess, "./training_files/tmp2/model.ckpt", global_step=self.model.global_step)
                                print("Model saved in path: %s" % save_path)
                                e += 1

                                json_stats = json.dumps(stats)
                                f = open("./training_files/losses.json","w")
                                f.write(json_stats)
                                f.close()
                        
                    finally:
                        save_path = self.model.saver.save(sess, "./training_files/tmp2/model.ckpt", global_step=self.model.global_step)
                        print("Model saved in path: %s" % save_path)
                        
                        json_stats = json.dumps(stats)
                        f = open("./training_files/losses.json","w")
                        f.write(json_stats)
                        f.close()

   