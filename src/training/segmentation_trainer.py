import tensorflow as tf
import argparse
import numpy as np
from tensorflow.python import debug as tf_debug

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


class SegmentationTrainer(Trainer):

    def train(self, restore=True, 
                    epochs=200, 
                    num_samples=None, 
                    training_per=0.8, 
                    random_seed=42, 
                    training=True, 
                    batch_size=1, 
                    save_steps=100,
                    start_epoch=0,
                    **kwargs):

        if self.dataset is None:
            if kwargs['segmentation_kitti']:
                self.dataset = KITTISegmentationDatasetLoader(self.data_base_path, num_samples, training_per, random_seed, training)
                self.eval_dataset = KITTISegmentationDatasetLoader(self.data_base_path, num_samples, training_per, random_seed, False)
            else:
                self.dataset = CityScapesSegmentationDatasetLoader(self.data_base_path, num_samples, training_per, random_seed, training)
                self.eval_dataset = CityScapesSegmentationDatasetLoader(self.data_base_path, num_samples, training_per, random_seed, False)

        with self.model.graph.as_default():
                
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as sess:
                if restore:
                    self.model.saver.restore(sess, tf.train.latest_checkpoint('./training_files/tmp_best2/'))
                else:
                    sess.run(tf.global_variables_initializer())

                self.__train_op(self.model.opt_img, sess, save_steps, kwargs['epochs_img_head'], start_epoch, **kwargs)
                self.__train_op(self.model.opt_img_all, sess, save_steps, kwargs['epochs_img_all'], start_epoch+kwargs['epochs_img_head'], **kwargs)


    def __train_op(self, opt, sess, save_steps, epochs, start_epoch, **kwargs):

        counter = 0
        for epoch in range(start_epoch, start_epoch+epochs, 1):
            print('Start epoch {0}'.format(epoch))

            self.dataset.reset_generator()

            losses = []
            accs = []

            try:
                # i = 0
                while True:
                            
                    images, labels = self.dataset.get_next()

                    loss, _, acc = sess.run([self.model.model_loss_img, opt, self.model.accuracy], 
                                                    feed_dict={self.model.train_inputs_rgb: images, self.model.y_true_img: labels,
                                                     self.model.train_inputs_lidar: np.zeros((1, 512, 448, 41)),
                                                     self.model.is_training: True, self.model.train_fusion_rgb: False})
                                
                    losses.append(loss)
                    accs.append(acc)
                    # i += 1
                    # if i % 10 == 0:
                    #     save_path = self.model.saver.save(sess, "./training_files/tmp/model.ckpt", global_step=self.model.global_step)
                    #     print("Model saved in path: %s" % save_path)

            except (tf.errors.OutOfRangeError, StopIteration):
                pass

            finally:
                save_path = self.model.saver.save(sess, "./training_files/tmp_best2/model.ckpt", global_step=self.model.global_step)
                print("Model saved in path: %s" % save_path)
                print('Loss:', np.mean(losses), ', Acc:', np.mean(accs))
                self.__save_summary(sess, losses, accs, epoch, True)
                self.eval(sess, epoch, **kwargs)
                losses = []
                accs = []

    def eval(self, sess, epoch, batch_size=1, **kwargs):
        self.eval_dataset.reset_generator()
        losses = []
        accs = []
        try:
            while True:
                            
                images, labels = self.eval_dataset.get_next()

                loss, acc = sess.run([self.model.model_loss_img, self.model.accuracy], 
                                                    feed_dict={self.model.train_inputs_rgb: images, self.model.y_true_img: labels,
                                                    self.model.train_inputs_lidar: np.zeros((1, 512, 448, 41)),\
                                                     self.model.is_training: False, self.model.train_fusion_rgb: False})
                                
                losses.append(loss)
                accs.append(acc)

        except (tf.errors.OutOfRangeError, StopIteration):
            pass

        finally:
            self.__save_summary(sess, losses, accs, epoch, False)
            print('Validation - Epoch {0}: Loss = {1}, Accuracy = {2}'.format(epoch, np.mean(np.array(losses)), np.mean(np.array(accs))))

        self.eval_dataset.reset_generator()
        images_cars = []
        images_road = []
        for i in range(kwargs['num_summary_images']):
            images, _ = self.eval_dataset.get_next()
            output = sess.run(self.model.detection_layer, feed_dict={self.model.train_inputs_rgb: images, 
                            self.model.train_inputs_lidar: np.zeros((1, 512, 448, 41)), self.model.is_training: False, self.model.train_fusion_rgb: False})
            output = output[0]

            images_cars.append(output[:, :, 0])
            images_road.append(output[:, :, 1])
        images_cars = np.array(images_cars).reshape((kwargs['num_summary_images'], 24, 78, 1))
        images_road = np.array(images_road).reshape((kwargs['num_summary_images'], 24, 78, 1))
        s = sess.run(self.model.images_summary_segmentation_cars, feed_dict={self.model.images_summary_segmentation_cars_placeholder: images_cars})
        self.model.validation_writer.add_summary(s, epoch)
        s = sess.run(self.model.images_summary_segmentation_road, feed_dict={self.model.images_summary_segmentation_road_placeholder: images_road})
        self.model.validation_writer.add_summary(s, epoch)




            


    def __save_summary(self, sess, epoch_loss, epoch_acc, epoch, training=True):
        model_loss_summary, acc_summary = sess.run([self.model.model_loss_image_summary, self.model.accuracy_image_summary], 
                                                                        {self.model.model_loss_image_summary_placeholder: np.mean(np.array(epoch_loss)),
                                                                         self.model.accuracy_image_summary_placeholder: np.mean(np.array(epoch_acc))})

        if training:
            writer = self.model.train_writer
        else:
            writer = self.model.validation_writer

        writer.add_summary(model_loss_summary, epoch)
        writer.add_summary(acc_summary, epoch)
