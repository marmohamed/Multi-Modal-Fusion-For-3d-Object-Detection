# import tensorflow as tf
# import argparse
# import numpy as np
# from tensorflow.python import debug as tf_debug
# import math

# from utils.constants import *
# from utils.utils import *

# from loss.losses import *

# from models.ResNetBuilder import *
# from models.ResnetImage import *

# from FPN.FPN import *

# from data.img_branch_data_reader import *


# class ImageDetectionModel():

#     def __init__(self, model_name='resnet', img_size_1=370, img_size_2=1224, c_dim=3, scale=32, lr=3e-3, th=0.5, res_blocks=3):
#         super().__init__()
#         self.CONST = Const()
#         self.model_name = model_name
#         self.build_model(img_size_1=img_size_1, img_size_2=img_size_2, c_dim=c_dim, scale=scale, lr=lr, th=th, res_blocks=res_blocks)

#     def build_model(self, img_size_1, img_size_2, c_dim, scale, lr, th, res_blocks):
#         self.graph = tf.Graph()
#         with self.graph.as_default():
#             self.train_inputs_rgb = tf.placeholder(tf.float32, 
#                                                         [None, img_size_1, img_size_2, c_dim], 
#                                                         name='train_inputs_rgb')
#             self.y_true = tf.placeholder(tf.float32, shape=(None, 48, 156, 1)) # target

#             self.last_features_layer = None     

#             self.cnn = ResNetBuilder().build(branch=self.CONST.IMAGE_BRANCH, img_height=img_size_1, img_width=img_size_2, img_channels=c_dim)
#             self.cnn.build_model(self.train_inputs_rgb)
#             with tf.variable_scope("image_head"):
#                 self.fpn_images = FPN(self.cnn.res_groups, "fpn_rgb_train")

#                 self.last_features_layer = self.fpn_images[1]

#                 for i in range(res_blocks):
#                     self.last_features_layer = resblock(self.last_features_layer, 128, scope='fpn_res_'+str(i))

#                 self.detection_layer = conv(self.last_features_layer, 1, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out')

#             self.model_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_true, logits=self.detection_layer))
            
#             head_only_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
#                                         "image_head")

#             self.global_step = tf.Variable(0, name='global_step', trainable=False)
#             self.decay_rate = tf.train.exponential_decay(lr, self.global_step, 200, 0.96, False)  
#             self.opt = tf.train.AdamOptimizer(self.decay_rate ).minimize(self.model_loss, var_list=head_only_vars, global_step=self.global_step)

#             self.equality = tf.where(self.y_true >= 0.5, tf.equal(tf.cast(tf.sigmoid(self.detection_layer) >= th, tf.float32), self.y_true), tf.zeros_like(self.y_true, dtype=tf.bool))
#             self.accuracy = tf.reduce_sum(tf.cast(self.equality, tf.float32)) / tf.cast(tf.count_nonzero(self.y_true), tf.float32)

#             self.equality_neg = tf.where(self.y_true < 0.5, tf.equal(tf.cast(tf.sigmoid(self.detection_layer) < th, tf.float32), self.y_true), tf.zeros_like(self.y_true, dtype=tf.bool))
#             self.accuracy_neg = tf.reduce_sum(tf.cast(self.equality_neg, tf.float32)) / tf.cast(tf.count_nonzero(1 - self.y_true), tf.float32)

#             self.saver = tf.train.Saver()
#             tf.summary.scalar('learning_rate', tf.squeeze(self.decay_rate))
#             tf.summary.scalar('accuracy', tf.squeeze(self.accuracy))
#             tf.summary.scalar('accuracy_negative', tf.squeeze(self.accuracy_neg))
#             tf.summary.scalar('model_loss', self.model_loss)

#             self.merged = tf.summary.merge_all()
#             self.train_writer = tf.summary.FileWriter('./train', self.graph)
#             self.test_writer = tf.summary.FileWriter('./test')
        

#     def train(self, base_path, restore=False, epochs=10, num_samples=None, training_per=0.8, random_seed=42, training=True, batch_size=4, save_steps=100):
#         with self.graph.as_default():
                
#             config = tf.ConfigProto()
#             config.gpu_options.allow_growth = True

#             with tf.Session(config=config) as sess:
#                 if restore:
#                     self.saver.restore(sess, tf.train.latest_checkpoint('./tmp/'))
#                 else:
#                     sess.run(tf.global_variables_initializer())

#                 counter = 0
#                 for epoch in range(epochs):
#                     print('Start epoch {0}'.format(epoch))

#                     dataset = make_dataset(base_path, batch_size=batch_size, num_samples=num_samples, training_per=training_per, random_seed=random_seed, training=training)
#                     losses = []
#                     accs = []
#                     accs_neg = []
#                     try:
#                         while True:
#                             images = []
#                             labels = []
#                             for _ in range(batch_size):
#                                 image, label = list(next(dataset))
#                                 images.extend(image)
#                                 labels.extend(label)

#                             images = np.array(images)
#                             labels = np.array(labels)

#                             loss, _, summary, acc, acc_neg = sess.run([self.model_loss, self.opt, self.merged, self.accuracy, self.accuracy_neg], 
#                                                     feed_dict={self.train_inputs_rgb: images, self.y_true: labels})
                                
#                             self.train_writer.add_summary(summary, counter)
#                             losses.append(loss)
#                             accs.append(acc)
#                             accs_neg.append(acc_neg)
#                             counter += 1
#                             if counter % save_steps == 0:
#                                 save_path = self.saver.save(sess, "./tmp/model.ckpt", global_step=self.global_step)
#                                 print("Model saved in path: %s" % save_path)
#                                 print('Loss:', np.mean(losses), ', Acc:', np.mean(accs), ', Neg acc:', np.mean(accs_neg))
#                                 losses = []
#                                 accs = []
#                                 accs_neg = []

#                     except tf.errors.OutOfRangeError:
#                         save_path = self.saver.save(sess, "./tmp/model.ckpt", global_step=self.global_step)
#                         print("Model saved in path: %s" % save_path)
#                         print('Loss:', np.mean(losses), ', Acc:', np.mean(accs), ', Neg acc:', np.mean(accs_neg))
#                         losses = []
#                         accs = []
#                         accs_neg = []

#                     except StopIteration:
#                         save_path = self.saver.save(sess, "./tmp/model.ckpt", global_step=self.global_step)
#                         print("Model saved in path: %s" % save_path)
#                         print('Loss:', np.mean(losses), ', Acc:', np.mean(accs), ', Neg acc:', np.mean(accs_neg))
#                         losses = []
#                         accs = []
#                         accs_neg = []

#                     save_path = self.saver.save(sess, "./tmp/model.ckpt", global_step=self.global_step)
#                     print("Model saved in path: %s" % save_path)
#                     print('Loss:', np.mean(losses), ', Acc:', np.mean(accs), ', Neg acc:', np.mean(accs_neg))
#                     losses = []
#                     accs = []
#                     accs_neg = []

#     def eval(self, base_path, num_samples=None, training_per=0.8, random_seed=42, training=False, batch_size=4):
#         with self.graph.as_default():
                
#             config = tf.ConfigProto()
#             config.gpu_options.allow_growth = True

#             with tf.Session(config=config) as sess:
#                 self.saver.restore(sess, tf.train.latest_checkpoint('./tmp/'))
                
#                 counter = 0
#                 print('Start epoch {0}'.format(epoch))

#                 dataset = make_dataset(base_path, batch_size=batch_size, num_samples=num_samples, training_per=training_per, random_seed=random_seed, training=training)
#                 losses = []
#                 accs = []
#                 accs_neg = []
#                 try:
#                     while True:
#                         images = []
#                         labels = []
#                         for _ in range(batch_size):
#                             image, label = list(next(dataset))
#                             images.append(image)
#                             labels.append(label)

#                         images = np.array(images)
#                         labels = np.array(labels)

#                         loss, summary, acc, acc_neg = sess.run([self.model_loss, self.merged, self.accuracy, self.accuracy_neg], 
#                                                 feed_dict={self.train_inputs_rgb: images, self.y_true: labels})
                            
#                         losses.append(loss)
#                         accs.append(acc)
#                         accs_neg.append(acc_neg)
#                         self.test_writer.add_summary(summary, counter)
                        
#                 except tf.errors.OutOfRangeError:
#                     pass
#                 except StopIteration:
#                     pass
                
#                 return losses, accs, accs_neg
    


                


