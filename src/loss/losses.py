import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod, ABCMeta
from loss.focal_loss import *



class LossCalculator(object):

    def get_precision_recall(self, truth, predictions, i):

        target_tensor = truth[:, :, :, i, -1]
        pred_sigmoid = tf.math.sigmoid(predictions[:, :, :, i, -1])

       
        tp = tf.reduce_sum(pred_sigmoid * target_tensor)

        fp = tf.reduce_sum(pred_sigmoid * (1.-target_tensor))
                
        fn = tf.reduce_sum((1 - pred_sigmoid) * target_tensor)

        tn = tf.reduce_sum((1-pred_sigmoid) * (1-target_tensor))

        recall_neg = tn / (tn + fp + 1e-8)
        recall_pos = tp  / (tp + fn + 1e-8)

        recall_neg = recall_neg / (tf.math.count_nonzero(1-target_tensor, dtype=tf.float32) + 1e-8)
        recall_pos = recall_pos / (tf.math.count_nonzero(target_tensor, dtype=tf.float32) + 1e-8)

        return recall_neg, recall_pos


    def macro_double_soft_f1(self, truth, predictions, i):
        """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
        Use probability values instead of binary predictions.
        This version uses the computation of soft-F1 for both positive and negative class for each label.
        
        Args:
            y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
            
        Returns:
            cost (scalar Tensor): value of the cost function for the batch
        """
        y = truth[:, :, :, i, -1]
        y_hat = tf.math.sigmoid(predictions[:, :, :, i, -1])
        y = tf.cast(y, tf.float32)
        y_hat = tf.cast(y_hat, tf.float32)
        tp = tf.reduce_sum(y_hat * y)
        fp = tf.reduce_sum(y_hat * (1 - y))
        fn = tf.reduce_sum((1 - y_hat) * y)
        tn = tf.reduce_sum((1 - y_hat) * (1 - y))
        soft_f1_class1 = tp / (tp + fn + fp + 1e-16)
        soft_f1_class0 = tn / (tn + fp + fn + 1e-16)
        # cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
        # cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
        # cost = 0.5 * (cost_class1 + cost_class0) # take into account both class 1 and class 0
        macro_cost_1 = 1 - (tf.reduce_sum(soft_f1_class1) / (tf.reduce_sum(y) + 1e-8))
        macro_cost_0 = 1 - (tf.reduce_sum(soft_f1_class0) / (tf.reduce_sum(1 - y) + 1e-8))
        # macro_cost_1 = tf.reduce_mean(macro_cost_1)
        # macro_cost_0 = tf.reduce_mean(macro_cost_0) 
        # macro_cost = tf.reduce_mean(cost) # average on all labels
        return macro_cost_0, macro_cost_1

    

    def __call__(self, truth, predictions, cls_loss, reg_loss, **params):

        # Classification
        c1 = cls_loss(truth[:, :, :, 0, 8], predictions[:, :, :, 0, 8], **params)
        c2 = cls_loss(truth[:, :, :, 1, 8], predictions[:, :, :, 1, 8], **params)
        classification_loss = tf.add_n([c1, c2])

        mask_true = tf.cast(tf.greater_equal(truth[:, :, :, :, -1],0.5), tf.int8)
        mask_not_true = tf.cast(tf.less(truth[:, :, :, :, -1],0.5), tf.int8)
        mask_pred = tf.cast(tf.greater_equal(tf.math.sigmoid(predictions[:, :, :, :, -1]),0.5), tf.int8)
        mask_not_pred = tf.cast(tf.less(tf.math.sigmoid(predictions[:, :, :, :, -1]),0.5), tf.int8)

        masks_and = tf.cast(tf.bitwise.bitwise_and(mask_true, mask_pred), tf.float32)
        tp = tf.math.count_nonzero(masks_and, dtype=tf.float32)

        masks_and_2 = tf.cast(tf.bitwise.bitwise_and(mask_not_true, mask_pred), tf.float32)
        fp = tf.math.count_nonzero(masks_and_2, dtype=tf.float32)

        masks_and_3 = tf.cast(tf.bitwise.bitwise_and(mask_true, mask_not_pred), tf.float32)
        fn = tf.math.count_nonzero(masks_and_3, dtype=tf.float32)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        
        recall_neg_0, recall_pos_0 = self.macro_double_soft_f1(truth, predictions, 0)
        recall_neg_1, recall_pos_1 = self.macro_double_soft_f1(truth, predictions, 1)
        recall_neg = recall_neg_0 + recall_neg_1
        recall_pos = recall_pos_0 + recall_pos_1

    
        #regression

        loss_fn = lambda t, p: tf.where(tf.greater_equal(truth[:, :, :, :, 8],0.5), reg_loss(t, p), tf.zeros_like(p))

        loc_ratios = np.array([2.4375, 1., 9.375*10 ])
        reg_losses1 = [loss_fn(truth[:, :, :, :, i], predictions[:, :, :, :, i]) * loc_ratios[i] for i in range(3)] 
        reg_losses2 = [loss_fn(truth[:, :, :, :, i], predictions[:, :, :, :, i]) for i in range(3, 6)] 
        reg_losses3 = [loss_fn((truth[:, :, :, :, i] + np.pi/4) / (np.pi/2), tf.math.sigmoid(predictions[:, :, :, :, i])) for i in range(6, 7)]

        loss_fn = lambda t, p: tf.where(tf.greater_equal(truth[:, :, :, :, 8],0.5), tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=p), tf.zeros_like(p))
        reg_losses4 = [loss_fn(truth[:, :, :, :, i], predictions[:, :, :, :, i]) for i in range(7, 8)]

        loc_reg_loss = tf.reduce_sum(reg_losses1) / (tf.math.count_nonzero(truth[:, :, :, :, 8], dtype=tf.float32)+1e-8)
        dim_reg_loss = tf.reduce_sum(reg_losses2) / (tf.math.count_nonzero(truth[:, :, :, :, 8], dtype=tf.float32)+1e-8)
        theta_reg_loss = tf.reduce_sum(reg_losses3) / (tf.math.count_nonzero(truth[:, :, :, :, 8], dtype=tf.float32)+1e-8)
        dir_reg_loss = tf.reduce_sum(reg_losses4) / (tf.math.count_nonzero(truth[:, :, :, :, 8], dtype=tf.float32)+1e-8)



        anchors_size=np.array([3.9, 1.6, 1.5])

        x = truth[:, :, :, :, 0]*anchors_size[0] + 0.5
        x_ = predictions[:, :, :, :, 0] * anchors_size[0] + 0.5
        y = truth[:, :, :, :, 1]*anchors_size[1] + 0.5
        y_ = predictions[:, :, :, :, 1]*anchors_size[1] + 0.5
        z = truth[:, :, :, :, 2]*anchors_size[2] + 1.
        z_ = predictions[:, :, :, :, 2]*anchors_size[2] + 1.

        size_true = tf.math.exp(truth[:, :, :, :, 3:6])*anchors_size
        size_pred = tf.math.exp(predictions[:, :, :, :, 3:6])*anchors_size


        x1 = x + size_true[:, :, :, :, 0]/2
        x2 = x - size_true[:, :, :, :, 0]/2
        y1 = y + size_true[:, :, :, :, 1]/2
        y2 = y - size_true[:, :, :, :, 1]/2
        z1 = z
        z2 = z - size_true[:, :, :, :, 2]

        x1_ = x_ + size_pred[:, :, :, :, 0]/2
        x2_ = x_ - size_pred[:, :, :, :, 0]/2
        y1_ = y_ + size_pred[:, :, :, :, 1]/2
        y2_ = y_ - size_pred[:, :, :, :, 1]/2
        z1_ = z_
        z2_ = z_ - size_pred[:, :, :, :, 2]

        area_g = size_true[:, :, :, :, 0] * size_true[:, :, :, :, 1]
        area_d = size_pred[:, :, :, :, 0] * size_pred[:, :, :, :, 1]
        h_g = size_true[:, :, :, :, 2]
        h_d = size_pred[:, :, :, :, 2]

        area_overlap = tf.maximum(0., (tf.minimum(x1, x1_) - tf.maximum(x2, x2_))) * tf.maximum(0., (tf.minimum(y1, y1_) - tf.maximum(y2, y2_)))
        h_overlap = tf.maximum(0., tf.minimum(z1, z1_) - tf.maximum(z2, z2_))

        iou = (area_overlap * h_overlap) / (area_g*h_g + area_d*h_d - area_overlap*h_overlap + 1e-8)
        iou = tf.where(tf.greater_equal(iou, 0), iou, tf.zeros_like(truth[:, :, :, :, 8]))
        iou = tf.where(tf.greater_equal(truth[:, :, :, :, 8],0.5), iou, tf.zeros_like(truth[:, :, :, :, 8]))
        iou = tf.reduce_sum(iou) / (tf.math.count_nonzero(truth[:, :, :, :, 8], dtype=tf.float32)+1e-8)


        x = truth[:, :, :, :, 0]*anchors_size[0] + 0.5
        x_ = predictions[:, :, :, :, 0]*anchors_size[0] + 0.5
        y = truth[:, :, :, :, 1]*anchors_size[1] + 0.5
        y_ = predictions[:, :, :, :, 1]*anchors_size[1] + 0.5
        z = truth[:, :, :, :, 2]*anchors_size[2] + 1.
        z_ = predictions[:, :, :, :, 2]*anchors_size[2] + 1.

        x1 = x + size_true[:, :, :, :, 0]/2
        x2 = x - size_true[:, :, :, :, 0]/2
        y1 = y + size_true[:, :, :, :, 1]/2
        y2 = y - size_true[:, :, :, :, 1]/2
        z1 = z
        z2 = z - size_true[:, :, :, :, 2]

        x1_ = x_ + size_true[:, :, :, :, 0]/2
        x2_ = x_ - size_true[:, :, :, :, 0]/2
        y1_ = y_ + size_true[:, :, :, :, 1]/2
        y2_ = y_ - size_true[:, :, :, :, 1]/2
        z1_ = z_
        z2_ = z_ - size_true[:, :, :, :, 2]

        area_g = size_true[:, :, :, :, 0] * size_true[:, :, :, :, 1]
        area_d = size_true[:, :, :, :, 0] * size_true[:, :, :, :, 1]
        h_g = size_true[:, :, :, :, 2]
        h_d = size_true[:, :, :, :, 2]

        area_overlap = tf.maximum(0., (tf.minimum(x1, x1_) - tf.maximum(x2, x2_))) * tf.maximum(0., (tf.minimum(y1, y1_) - tf.maximum(y2, y2_)))
        h_overlap = tf.maximum(0., tf.minimum(z1, z1_) - tf.maximum(z2, z2_))

        iou_loc = (area_overlap * h_overlap) / (area_g*h_g + area_d*h_d - area_overlap*h_overlap + 1e-8)
        iou_loc = tf.where(tf.greater_equal(iou_loc, 0), iou_loc, tf.zeros_like(truth[:, :, :, :, 8]))
        iou_loc = tf.where(tf.greater_equal(truth[:, :, :, :, 8],0.5), iou_loc, tf.zeros_like(truth[:, :, :, :, 8]))
        iou_loc = tf.reduce_sum(iou_loc) / (tf.math.count_nonzero(truth[:, :, :, :, 8], dtype=tf.float32)+1e-8)



        ##############################################


        x = truth[:, :, :, :, 0]*anchors_size[0] + 0.5
        x_ = predictions[:, :, :, :, 0] * anchors_size[0] + 0.5
        y = truth[:, :, :, :, 1]*anchors_size[1] + 0.5
        y_ = y
        z = truth[:, :, :, :, 2]*anchors_size[2] + 1.
        z_ = z

        x1 = x + size_true[:, :, :, :, 0]/2
        x2 = x - size_true[:, :, :, :, 0]/2
        y1 = y + size_true[:, :, :, :, 1]/2
        y2 = y - size_true[:, :, :, :, 1]/2
        z1 = z
        z2 = z - size_true[:, :, :, :, 2]

        x1_ = x_ + size_true[:, :, :, :, 0]/2
        x2_ = x_ - size_true[:, :, :, :, 0]/2
        y1_ = y_ + size_true[:, :, :, :, 1]/2
        y2_ = y_ - size_true[:, :, :, :, 1]/2
        z1_ = z_
        z2_ = z_ - size_true[:, :, :, :, 2]

        area_g = size_true[:, :, :, :, 0] * size_true[:, :, :, :, 1]
        area_d = size_true[:, :, :, :, 0] * size_true[:, :, :, :, 1]
        h_g = size_true[:, :, :, :, 2]
        h_d = size_true[:, :, :, :, 2]

        area_overlap = tf.maximum(0., (tf.minimum(x1, x1_) - tf.maximum(x2, x2_))) * tf.maximum(0., (tf.minimum(y1, y1_) - tf.maximum(y2, y2_)))
        h_overlap = tf.maximum(0., tf.minimum(z1, z1_) - tf.maximum(z2, z2_))

        iou_loc_x = (area_overlap * h_overlap) / (area_g*h_g + area_d*h_d - area_overlap*h_overlap + 1e-8)
        iou_loc_x = tf.where(tf.greater_equal(iou_loc_x, 0), iou_loc_x, tf.zeros_like(truth[:, :, :, :, 8]))
        iou_loc_x = tf.where(tf.greater_equal(truth[:, :, :, :, 8],0.5), iou_loc_x, tf.zeros_like(truth[:, :, :, :, 8]))
        iou_loc_x = tf.reduce_sum(iou_loc_x) / (tf.math.count_nonzero(truth[:, :, :, :, 8], dtype=tf.float32)+1e-8)




        x = truth[:, :, :, :, 0]*anchors_size[0] + 0.5
        x_ = x
        y = truth[:, :, :, :, 1]*anchors_size[1] + 0.5
        y_ = predictions[:, :, :, :, 1]*anchors_size[1] + 0.5
        z = truth[:, :, :, :, 2]*anchors_size[2] + 1.
        z_ = z

        x1 = x + size_true[:, :, :, :, 0]/2
        x2 = x - size_true[:, :, :, :, 0]/2
        y1 = y + size_true[:, :, :, :, 1]/2
        y2 = y - size_true[:, :, :, :, 1]/2
        z1 = z
        z2 = z - size_true[:, :, :, :, 2]

        x1_ = x_ + size_true[:, :, :, :, 0]/2
        x2_ = x_ - size_true[:, :, :, :, 0]/2
        y1_ = y_ + size_true[:, :, :, :, 1]/2
        y2_ = y_ - size_true[:, :, :, :, 1]/2
        z1_ = z_
        z2_ = z_ - size_true[:, :, :, :, 2]

        area_g = size_true[:, :, :, :, 0] * size_true[:, :, :, :, 1]
        area_d = size_true[:, :, :, :, 0] * size_true[:, :, :, :, 1]
        h_g = size_true[:, :, :, :, 2]
        h_d = size_true[:, :, :, :, 2]

        area_overlap = tf.maximum(0., (tf.minimum(x1, x1_) - tf.maximum(x2, x2_))) * tf.maximum(0., (tf.minimum(y1, y1_) - tf.maximum(y2, y2_)))
        h_overlap = tf.maximum(0., tf.minimum(z1, z1_) - tf.maximum(z2, z2_))

        iou_loc_y = (area_overlap * h_overlap) / (area_g*h_g + area_d*h_d - area_overlap*h_overlap + 1e-8)
        iou_loc_y = tf.where(tf.greater_equal(iou_loc_y, 0), iou_loc_y, tf.zeros_like(truth[:, :, :, :, 8]))
        iou_loc_y = tf.where(tf.greater_equal(truth[:, :, :, :, 8],0.5), iou_loc_y, tf.zeros_like(truth[:, :, :, :, 8]))
        iou_loc_y = tf.reduce_sum(iou_loc_y) / (tf.math.count_nonzero(truth[:, :, :, :, 8], dtype=tf.float32)+1e-8)



        x = truth[:, :, :, :, 0]*anchors_size[0] + 0.5
        x_ = x
        y = truth[:, :, :, :, 1]*anchors_size[1] + 0.5
        y_ = y
        z = truth[:, :, :, :, 2]*anchors_size[2] + 1.
        z_ = predictions[:, :, :, :, 2]*anchors_size[2] + 1.

        x1 = x + size_true[:, :, :, :, 0]/2
        x2 = x - size_true[:, :, :, :, 0]/2
        y1 = y + size_true[:, :, :, :, 1]/2
        y2 = y - size_true[:, :, :, :, 1]/2
        z1 = z
        z2 = z - size_true[:, :, :, :, 2]

        x1_ = x_ + size_true[:, :, :, :, 0]/2
        x2_ = x_ - size_true[:, :, :, :, 0]/2
        y1_ = y_ + size_true[:, :, :, :, 1]/2
        y2_ = y_ - size_true[:, :, :, :, 1]/2
        z1_ = z_
        z2_ = z_ - size_true[:, :, :, :, 2]

        area_g = size_true[:, :, :, :, 0] * size_true[:, :, :, :, 1]
        area_d = size_true[:, :, :, :, 0] * size_true[:, :, :, :, 1]
        h_g = size_true[:, :, :, :, 2]
        h_d = size_true[:, :, :, :, 2]

        area_overlap = tf.maximum(0., (tf.minimum(x1, x1_) - tf.maximum(x2, x2_))) * tf.maximum(0., (tf.minimum(y1, y1_) - tf.maximum(y2, y2_)))
        h_overlap = tf.maximum(0., tf.minimum(z1, z1_) - tf.maximum(z2, z2_))

        iou_loc_z = (area_overlap * h_overlap) / (area_g*h_g + area_d*h_d - area_overlap*h_overlap + 1e-8)
        iou_loc_z = tf.where(tf.greater_equal(iou_loc_z, 0), iou_loc_z, tf.zeros_like(truth[:, :, :, :, 8]))
        iou_loc_z = tf.where(tf.greater_equal(truth[:, :, :, :, 8],0.5), iou_loc_z, tf.zeros_like(truth[:, :, :, :, 8]))
        iou_loc_z = tf.reduce_sum(iou_loc_z) / (tf.math.count_nonzero(truth[:, :, :, :, 8], dtype=tf.float32)+1e-8)


        #################################################


        x = truth[:, :, :, :, 0]*anchors_size[0] + 0.5
        y = truth[:, :, :, :, 1]*anchors_size[1] + 0.5
        z = truth[:, :, :, :, 2]*anchors_size[2] + 1.

        size_true = tf.math.exp(truth[:, :, :, :, 3:6])*anchors_size
        size_pred = tf.math.exp(predictions[:, :, :, :, 3:6])*anchors_size


        x1 = x + size_true[:, :, :, :, 0]/2
        x2 = x - size_true[:, :, :, :, 0]/2
        y1 = y + size_true[:, :, :, :, 1]/2
        y2 = y - size_true[:, :, :, :, 1]/2
        z1 = z
        z2 = z - size_true[:, :, :, :, 2]

        x1_ = x + size_pred[:, :, :, :, 0]/2
        x2_ = x - size_pred[:, :, :, :, 0]/2
        y1_ = y + size_pred[:, :, :, :, 1]/2
        y2_ = y - size_pred[:, :, :, :, 1]/2
        z1_ = z
        z2_ = z - size_pred[:, :, :, :, 2]

        area_g = size_true[:, :, :, :, 0] * size_true[:, :, :, :, 1]
        area_d = size_pred[:, :, :, :, 0] * size_pred[:, :, :, :, 1]
        h_g = size_true[:, :, :, :, 2]
        h_d = size_pred[:, :, :, :, 2]

        area_overlap = tf.maximum(0., (tf.minimum(x1, x1_) - tf.maximum(x2, x2_))) * tf.maximum(0., (tf.minimum(y1, y1_) - tf.maximum(y2, y2_)))
        h_overlap = tf.maximum(0., tf.minimum(z1, z1_) - tf.maximum(z2, z2_))

        iou_dim = (area_overlap * h_overlap) / (area_g*h_g + area_d*h_d - area_overlap*h_overlap + 1e-8)
        iou_dim = tf.where(tf.greater_equal(iou_dim, 0), iou_dim, tf.zeros_like(truth[:, :, :, :, 8]))
        iou_dim = tf.where(tf.greater_equal(truth[:, :, :, :, 8],0.5), iou_dim, tf.zeros_like(truth[:, :, :, :, 8]))
        iou_dim = tf.reduce_sum(iou_dim) / (tf.math.count_nonzero(truth[:, :, :, :, 8], dtype=tf.float32)+1e-8)



        # theta_pred = (tf.math.sigmoid(predictions[:, :, :, :, 6]) * np.pi/2) * 57.2958
        # theta_truth = (truth[:, :, :, :, 6] + np.pi/4) * 57.2958
        # theta_diff = tf.abs(theta_pred-theta_truth)
        # mask_theta = tf.cast(tf.equal(theta_diff, 0), tf.float32)
        # mask_theta = tf.where(tf.greater_equal(truth[:, :, :, :, 8],0.5), mask_theta, tf.zeros_like(truth[:, :, :, :, 8]))
        # true_count_theta = tf.math.count_nonzero(truth[:, :, :, :, -1], dtype=tf.float32)
        # pred_count_theta = tf.math.count_nonzero(mask_theta, dtype=tf.float32)
        # accuracy_theta = pred_count_theta / (true_count_theta + 1e-8)
        accuracy_theta = 1
        

        theta_pred = (tf.math.sigmoid(predictions[:, :, :, :, 6]) * np.pi/2) * 57.2958
        theta_truth = (truth[:, :, :, :, 6] + np.pi/4) * 57.2958
        theta_diff = tf.abs(theta_pred-theta_truth)
        theta_diff = tf.where(tf.greater_equal(truth[:, :, :, :, 8],0.5), theta_diff, tf.zeros_like(truth[:, :, :, :, 8]))
        true_count_theta = tf.math.count_nonzero(truth[:, :, :, :, -1], dtype=tf.float32)
        accuracy_theta = tf.reduce_sum(theta_diff) / (true_count_theta + 1e-8)
   
        return classification_loss,\
                loc_reg_loss,\
                dim_reg_loss,\
                theta_reg_loss,\
                dir_reg_loss,\
                precision, recall, iou, iou_loc, iou_dim, accuracy_theta,\
                recall_pos, recall_neg,\
                iou_loc_x, iou_loc_y, iou_loc_z
                
 

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

