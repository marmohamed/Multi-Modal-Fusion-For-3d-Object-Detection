import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod, ABCMeta
from loss.focal_loss import *
from loss.metrics_utils import *
from loss.iou_utils import *
from loss.corners_loss import *



class LossCalculator(object):

    def __init__(self):
        self.metrics_helper = MetricsHelper()
        self.iou_helper = IOUHelper()


    def __call__(self, truth, predictions, cls_loss, reg_loss, **params):

        # Classification
        classification_loss = cls_loss(truth[:, :, :, :, 8], predictions[:, :, :, :, 8], **params)

        precision, recall = self.get_precision_recall(truth, predictions)

        #regression

        loss_fn = lambda t, p: tf.where(tf.greater_equal(truth[:, :, :, :, 8],0.5), reg_loss(t, p), tf.zeros_like(p))

        reg_losses1 = [loss_fn(truth[:, :, :, :, i], tf.math.sigmoid(predictions[:, :, :, :, i])-0.5) for i in range(3)] 
        reg_losses2 = [loss_fn(truth[:, :, :, :, i], tf.nn.tanh(predictions[:, :, :, :, i])) for i in range(3, 6)] 
        # reg_losses3 = [loss_fn(truth[:, :, :, :, i] , predictions[:, :, :, :, i]) for i in range(6, 8)]
        reg_losses3 = [loss_fn((truth[:, :, :, :, i] + np.pi/4) / (np.pi/2), tf.math.sigmoid(predictions[:, :, :, :, i])) for i in range(6, 7)]
        loss_fn = lambda t, p: tf.where(tf.greater_equal(truth[:, :, :, :, 8],0.5), tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=p), tf.zeros_like(p))
        reg_losses4 = [loss_fn(truth[:, :, :, :, i], predictions[:, :, :, :, i]) for i in range(7, 8)]

        c = (tf.math.count_nonzero(truth[:, :, :, :, 8], dtype=tf.float32)+1e-8)

        loc_reg_loss = tf.reduce_sum(reg_losses1)  / c
        dim_reg_loss = tf.reduce_sum(reg_losses2) / c
        theta_reg_loss = tf.reduce_sum(reg_losses3) / c
        dir_reg_loss = tf.reduce_sum(reg_losses4) / c

        iou, iou_2d = self.get_iou(truth, predictions)
        iou_dim = self.get_iou_dim(truth, predictions)
        iou_loc = self.get_iou_loc(truth, predictions)

        accuracy_theta = self.get_accracy_diffs(truth, predictions)
        
        corners_losses = get_corners_loss(truth, predictions)
        corners_loss = tf.where(tf.greater_equal(truth[:, :, :, :, 8],0.5), corners_losses, tf.zeros_like(truth[:, :, :, :, 8]))
        corners_loss = tf.reduce_sum(corners_loss) / c


        oclussion_loss_fn = lambda t, p: tf.where(tf.greater_equal(truth[:, :, :, :, 8],0.5),  tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=p), tf.zeros_like(truth[:, :, :, :, 8]))
        oclussion_loss = oclussion_loss_fn(truth[:, :, :, :, 9:13], predictions[:, :, :, :, 9:13])
        oclussion_loss = tf.reduce_sum(oclussion_loss)/c

        truth_height = tf.expand_dims(tf.math.reduce_max(truth[:, :, :, :, 8], axis=1), axis=1)
        corners_losses_height = get_corners_loss_height(truth, predictions)
        corners_loss_height = tf.where(tf.greater_equal(truth_height,0.5), corners_losses_height, tf.zeros_like(truth_height))
        
        corners_loss_height = tf.reduce_sum(corners_loss_height) / c

        corners_loss_all = corners_loss + corners_loss_height
   
        return classification_loss,\
                loc_reg_loss,\
                dim_reg_loss,\
                theta_reg_loss,\
                dir_reg_loss,\
                corners_loss,\
                corners_loss_all,\
                precision, recall, iou, iou_2d, iou_loc, iou_dim, accuracy_theta


    def get_precision_recall_loss(self, truth, predictions):
        recall_neg, recall_pos = self.metrics_helper.get_precision_recall_loss(truth, predictions)
        return recall_neg, recall_pos

    def get_precision_recall(self, truth, predictions):
        precision, recall = self.metrics_helper.get_precision_recall(truth, predictions)
        return precision, recall

    def macro_double_soft_f1(self, truth, predictions):
        macro_neg_cost_0, macro_pos_cost_0 = self.metrics_helper.macro_double_soft_f1(truth, predictions)
        return macro_neg_cost_0+macro_pos_cost_0, macro_neg_cost_0+macro_pos_cost_0
                
    def get_accracy_diffs(self, truth, predictions):
        return self.metrics_helper.get_accracy_diffs(truth, predictions)

    def get_iou(self, truth, predictions):
        iou, iou_2d = self.iou_helper.get_iou(truth, predictions)
        return iou, iou_2d

    def get_iou_loc(self, truth, predictions):
        return self.iou_helper.get_iou_loc(truth, predictions)

    def get_iou_dim(self, truth, predictions):
        return self.iou_helper.get_iou_dim(truth, predictions)

    def get_iou_loc_x(self, truth, predictions):
        return self.iou_helper.get_iou_loc_x(truth, predictions)

    def get_iou_loc_y(self, truth, predictions):
        return self.iou_helper.get_iou_loc_y(truth, predictions)

    def get_iou_loc_z(self, truth, predictions):
        return self.iou_helper.get_iou_loc_z(truth, predictions)


class Loss(object):

    __metaclass__ = ABCMeta

    def __init__(self, scope):
        # super(ABC, self).__init__()
        self.scope = scope

    def __call__(self, truth, predictions, **params):
        with tf.variable_scope(self.scope):
            return self._compute_loss(truth, predictions, **params)

    @abstractmethod
    def _compute_loss(self, truth, predictions, **params):
        pass


class RegLoss(Loss):

    def _compute_loss(self, truth, predictions, **params):
        if 'mse_loss' in params and params['mse_loss']:
            out_loss = tf.losses.mean_squared_error(labels=truth, predictions=predictions)
        else:
            sub_value = tf.abs(tf.subtract(predictions, truth))
            loss1 = sub_value - 0.5
            loss2 = tf.square(sub_value) * 0.5
            out_loss = tf.where(tf.greater_equal(sub_value, 1), loss1, loss2)
        return out_loss


class ClsLoss(Loss):

    def _compute_loss(self, truth, predictions, **params):
        self.cls_loss_out = None
        if 'focal_loss' in params and params['focal_loss']:
            temp = focal_loss(predictions, truth, weights=None, alpha=0.25, gamma=2)
            return temp
        else:
            temp = tf.nn.sigmoid_cross_entropy_with_logits(labels=truth, logits=predictions)
            temp = tf.reduce_mean(temp)
            return temp