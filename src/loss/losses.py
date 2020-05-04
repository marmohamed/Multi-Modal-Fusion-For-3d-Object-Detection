import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod, ABCMeta
from loss.focal_loss import *



class LossCalculator(object):

    def get_precision_recall(self, truth, predictions, i):

        target_tensor = 1-truth[:, :, :, i, -1]
        pred_sigmoid = 1-tf.math.sigmoid(predictions[:, :, :, i, -1])

        zeros = array_ops.zeros_like(pred_sigmoid, dtype=pred_sigmoid.dtype)
       

        pred_sigmoid_x_true = pred_sigmoid * target_tensor
        pred_sigmoid_x_true2_prob2 =  tf.sign(pred_sigmoid_x_true)
        tp = pred_sigmoid_x_true
        tp = tf.reduce_sum(tp)
        tp_c = tf.reduce_sum(pred_sigmoid_x_true2_prob2)

        fp = pred_sigmoid * (1.-target_tensor) 
        fp = tf.reduce_sum(fp)
                
        fn = (1 - pred_sigmoid) * target_tensor
        fn = tf.reduce_sum(fn)

        precision = tp / (tp_c + fp + 1e-8)
        recall = tp  / (tp_c + fn + 1e-8)

        return precision, recall

    def __call__(self, truth, predictions, cls_loss, reg_loss, **params):

        precision = 0
        recall = 0

        for i in range(2):

            # pred_sigmoid = tf.math.sigmoid(predictions[:, :, :, i, -1])
            # pred_sigmoid_x_true = pred_sigmoid * truth[:, :, :, i, -1]
            # pred_sigmoid_x_not_true = pred_sigmoid * (1-truth[:, :, :, i, -1])


            # zeros = array_ops.zeros_like(pred_sigmoid, dtype=pred_sigmoid.dtype)
            # ones = array_ops.ones_like(pred_sigmoid, dtype=pred_sigmoid.dtype)

            # pred_sigmoid_x_true2 = pred_sigmoid_x_true - 0
            # pred_sigmoid_x_true2_prob = array_ops.where(pred_sigmoid_x_true2 > zeros, pred_sigmoid, zeros) 
            # pred_sigmoid_x_true2_prob2 = array_ops.where(pred_sigmoid_x_true2 > zeros, tf.sign(pred_sigmoid), zeros) 
            # tp = tf.reduce_sum(pred_sigmoid_x_true2_prob2)
            # tp2 = tf.reduce_sum(pred_sigmoid_x_true2_prob)


            # pred_sigmoid_x_not_true2 = pred_sigmoid_x_not_true - 0.
            # pred_sigmoid_x_not_true2_ = array_ops.where(pred_sigmoid_x_not_true2 > zeros, pred_sigmoid, zeros)
            # fp = tf.reduce_sum(pred_sigmoid_x_not_true2_)
            
            # pred_sigmoid_x_true3 = pred_sigmoid_x_true * truth[:, :, :, i, -1]
            # pred_sigmoid_x_true3_1 = 1. - pred_sigmoid_x_true3
            # pred_sigmoid_x_true3_2 = array_ops.where(pred_sigmoid_x_true3_1 < ones, pred_sigmoid_x_true3_1, zeros)
            # fn = tf.reduce_sum(pred_sigmoid_x_true3_2)

            # precision += tp2 / (tp + fp + 1e-8)
            # recall += tp2  / (tp + fn + 1e-8)
            precision_, recall_ = self.get_precision_recall(truth, predictions, i)
            precision += precision_
            recall += recall_

        # Classification
        c1 = cls_loss(truth[:, :, :, 0, 8], predictions[:, :, :, 0, 8], **params)
        c2 = cls_loss(truth[:, :, :, 1, 8], predictions[:, :, :, 1, 8], **params)
        classification_loss = tf.add_n([c1, c2])

        # mask_true = tf.cast(tf.greater_equal(truth[:, :, :, :, -1],0.5), tf.int8)
        # mask_not_true = tf.cast(tf.less(truth[:, :, :, :, -1],0.5), tf.int8)
        # mask_pred = tf.cast(tf.greater_equal(tf.math.sigmoid(predictions[:, :, :, :, -1]),0.5), tf.int8)
        # mask_not_pred = tf.cast(tf.less(tf.math.sigmoid(predictions[:, :, :, :, -1]),0.5), tf.int8)

        # masks_and = tf.cast(tf.bitwise.bitwise_and(mask_true, mask_pred), tf.float32)
        # tp = tf.math.count_nonzero(masks_and, dtype=tf.float32)

        # masks_and_2 = tf.cast(tf.bitwise.bitwise_and(mask_not_true, mask_pred), tf.float32)
        # fp = tf.math.count_nonzero(masks_and_2, dtype=tf.float32)

        # masks_and_3 = tf.cast(tf.bitwise.bitwise_and(mask_true, mask_not_pred), tf.float32)
        # fn = tf.math.count_nonzero(masks_and_3, dtype=tf.float32)

        # precision = tp / (tp + fp + 1e-8)
        # recall = tp / (tp + fn + 1e-8)

        

        #regression

        loss_fn = lambda t, p: tf.where(tf.greater_equal(truth[:, :, :, :, 8],0.5), reg_loss(t, p), tf.zeros_like(p))

        loc_ratios = np.array([2.4375, 1., 9.375 ])
        reg_losses1 = [loss_fn(truth[:, :, :, :, i], tf.tanh(predictions[:, :, :, :, i])*0.5) * loc_ratios[i] for i in range(3)] 
        reg_losses2 = [loss_fn(truth[:, :, :, :, i], tf.nn.relu(predictions[:, :, :, :, i])) for i in range(3, 6)] 
        reg_losses3 = [loss_fn(truth[:, :, :, :, i], tf.math.sigmoid(predictions[:, :, :, :, i]) * np.pi/2 - np.pi/4) for i in range(6, 7)]

        loss_fn = lambda t, p: tf.where(tf.greater_equal(truth[:, :, :, :, 8],0.5), tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=p), tf.zeros_like(p))
        reg_losses4 = [loss_fn(truth[:, :, :, :, i], predictions[:, :, :, :, i]) for i in range(7, 8)]

        loc_reg_loss = tf.reduce_sum(reg_losses1) / (tf.math.count_nonzero(truth[:, :, :, :, 8], dtype=tf.float32)+1e-8)
        dim_reg_loss = tf.reduce_sum(reg_losses2) / (tf.math.count_nonzero(truth[:, :, :, :, 8], dtype=tf.float32)+1e-8)
        theta_reg_loss = tf.reduce_sum(reg_losses3) / (tf.math.count_nonzero(truth[:, :, :, :, 8], dtype=tf.float32)+1e-8)
        dir_reg_loss = tf.reduce_sum(reg_losses4) / (tf.math.count_nonzero(truth[:, :, :, :, 8], dtype=tf.float32)+1e-8)



        anchors_size=np.array([3.9, 1.6, 1.5])

        x = truth[:, :, :, :, 0]*anchors_size[0] + 0.5
        x_ = tf.tanh(predictions[:, :, :, :, 0])*0.5*anchors_size[0] + 0.5
        y = truth[:, :, :, :, 1]*anchors_size[1] + 0.5
        y_ = tf.tanh(predictions[:, :, :, :, 1])*0.5*anchors_size[1] + 0.5
        z = truth[:, :, :, :, 2]*anchors_size[2] + 0.5
        z_ = tf.tanh(predictions[:, :, :, :, 2])*0.5*anchors_size[2] + 0.5

        size_true = tf.math.square(truth[:, :, :, :, 3:6])*anchors_size
        size_pred = tf.math.square(tf.nn.relu(predictions[:, :, :, :, 3:6]))*anchors_size


        x1 = x + size_true[:, :, :, :, 0]/2
        x2 = x - size_true[:, :, :, :, 0]/2
        y1 = y + size_true[:, :, :, :, 1]/2
        y2 = y - size_true[:, :, :, :, 1]/2
        z1 = z + size_true[:, :, :, :, 2]/2
        z2 = z - size_true[:, :, :, :, 2]/2

        x1_ = x_ + size_pred[:, :, :, :, 0]/2
        x2_ = x_ - size_pred[:, :, :, :, 0]/2
        y1_ = y_ + size_pred[:, :, :, :, 1]/2
        y2_ = y_ - size_pred[:, :, :, :, 1]/2
        z1_ = z_ + size_pred[:, :, :, :, 2]/2
        z2_ = z_ - size_pred[:, :, :, :, 2]/2

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
        x_ = tf.tanh(predictions[:, :, :, :, 0])*0.5*anchors_size[0] + 0.5
        y = truth[:, :, :, :, 1]*anchors_size[1] + 0.5
        y_ = tf.tanh(predictions[:, :, :, :, 1])*0.5*anchors_size[1] + 0.5
        z = truth[:, :, :, :, 2]*anchors_size[2] + 0.5
        z_ = tf.tanh(predictions[:, :, :, :, 2])*0.5*anchors_size[2] + 0.5

        x1 = x + size_true[:, :, :, :, 0]/2
        x2 = x - size_true[:, :, :, :, 0]/2
        y1 = y + size_true[:, :, :, :, 1]/2
        y2 = y - size_true[:, :, :, :, 1]/2
        z1 = z + size_true[:, :, :, :, 2]/2
        z2 = z - size_true[:, :, :, :, 2]/2

        x1_ = x_ + size_true[:, :, :, :, 0]/2
        x2_ = x_ - size_true[:, :, :, :, 0]/2
        y1_ = y_ + size_true[:, :, :, :, 1]/2
        y2_ = y_ - size_true[:, :, :, :, 1]/2
        z1_ = z_ + size_true[:, :, :, :, 2]/2
        z2_ = z_ - size_true[:, :, :, :, 2]/2

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



        x = truth[:, :, :, :, 0]*anchors_size[0] + 0.5
        y = truth[:, :, :, :, 1]*anchors_size[1] + 0.5
        z = truth[:, :, :, :, 2]*anchors_size[2] + 0.5

        size_true = tf.math.square(truth[:, :, :, :, 3:6])*anchors_size
        size_pred = tf.math.square(tf.nn.relu(predictions[:, :, :, :, 3:6]))*anchors_size


        x1 = x + size_true[:, :, :, :, 0]/2
        x2 = x - size_true[:, :, :, :, 0]/2
        y1 = y + size_true[:, :, :, :, 1]/2
        y2 = y - size_true[:, :, :, :, 1]/2
        z1 = z + size_true[:, :, :, :, 2]/2
        z2 = z - size_true[:, :, :, :, 2]/2

        x1_ = x + size_pred[:, :, :, :, 0]/2
        x2_ = x - size_pred[:, :, :, :, 0]/2
        y1_ = y + size_pred[:, :, :, :, 1]/2
        y2_ = y - size_pred[:, :, :, :, 1]/2
        z1_ = z + size_pred[:, :, :, :, 2]/2
        z2_ = z - size_pred[:, :, :, :, 2]/2

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



        theta_pred = (tf.math.sigmoid(predictions[:, :, :, :, 6]) * np.pi/2) * 57.2958
        theta_truth = (truth[:, :, :, :, 6] + np.pi/4) * 57.2958
        theta_diff = tf.abs(theta_pred-theta_truth)
        mask_theta = tf.cast(tf.equal(theta_diff, 0), tf.float32)
        mask_theta = tf.where(tf.greater_equal(truth[:, :, :, :, 8],0.5), mask_theta, tf.zeros_like(truth[:, :, :, :, 8]))
        true_count_theta = tf.math.count_nonzero(truth[:, :, :, :, -1], dtype=tf.float32)
        pred_count_theta = tf.math.count_nonzero(mask_theta, dtype=tf.float32)
        accuracy_theta = pred_count_theta / (true_count_theta + 1e-8)
        
        
        return classification_loss, precision, recall,\
                loc_reg_loss,\
                dim_reg_loss,\
                iou, iou_loc, iou_dim,\
                theta_reg_loss, accuracy_theta,\
                dir_reg_loss
                
 

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


# class IOULoss(Loss):

#     def roty(t):
#         ''' Rotation about the y-axis. '''
#         c = tf.math.cos(t)
#         s = tf.math.sin(t)
#         return tf.concat([[c,  0,  s],
#                         [0,  1,  0],
#                         [-s, 0,  c]])


#     def get_dims_util(xyz, hwl, theta):

#         R = roty(theta)

#         # 3d bounding box dimensions
#         h = hwl[:, :, :, 0]
#         w = hwl[:, :, :, 1]
#         l = hwl[:, :, :, 2]
        
#         # 3d bounding box corners
#         x_corners = tf.concat([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2])
#         y_corners = tf.concat([0,0,0,0,-h,-h,-h,-h])
#         z_corners = tf.concat([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2])

#         # rotate and translate 3d bounding box
#         corners_3d = R * tf.concat([x_corners,y_corners,z_corners]))
#         #print corners_3d.shape
#         corners_3d[:,:] = corners_3d[:,:] + xyz

#         return corners_3d
        

#     def get_dims(k, truth, predictions, input_size=(512, 448), output_size=(128, 112)):

#             posxy = []
#             for i in range(output_size[0]):
#                 temp = []
#                 for j in range(output_size[1]):
#                     temp.append([i+0.5, j+0.5])
#                 posxy.append(temp)
#             anchors_pos = tf.constant(posxy, dtype=tf.float32)
#             anchors=tf.constant([1.6, 3.9, 1.5], dtype=tf.float32)

#             i = k
#             xyz = truth[:, :, :, i, 0:4]
#             xyz = xyz * anchors + anchors_pos
#             xyz = xyz * tf.constant([4, 4, 32], dtype=tf.float32)

#             xyz_pred = predictions[:, :, :, i, 0:4]
#             xyz_pred = xyz_pred * anchors + anchors_pos
#             xyz_pred = xyz_pred * tf.constant([4, 4, 32], dtype=tf.float32)

#             hwl = truth[:, :, :, i, 3:6]
#             hwl = tf.exp(hwl) * anchors

#             hwl_pred = predictions[:, :, :, i, 3:6]
#             hwl_pred = tf.exp(hwl_pred) * anchors

#             theta = truth[:, :, :, i, 6]
#             if i == 0:
#                 theta = tf.where(theta<0, theta + np.pi, theta)

#             theta_pred = predictions[:, :, :, i, 6]
#             theta_pred = tf.math.sigmoid(theta_pred) * np.pi/2 - np.pi/4
#             if i == 0:
#                 theta_pred = tf.where(theta_pred<0, theta + np.pi, theta)

#             theta_pred = theta_pred + i * (np.pi/2)

#             return get_dims_util(xyz, hwl, theta), get_dims_util(xyz_pred, hwl_pred, theta_pred)





#     def _compute_loss(self, truth, predictions, **params):
#         t0 = get_dims(0, truth, predictions)
#         t1 = get_dims(1, truth, predictions)

#         return 0
        
