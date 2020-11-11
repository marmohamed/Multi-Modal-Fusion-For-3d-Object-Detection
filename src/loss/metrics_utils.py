
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod, ABCMeta



class MetricsHelper:

    def get_precision_recall_loss(self, truth, predictions):

        target_tensor = truth[:, :, :, -1]
        pred_sigmoid = tf.math.sigmoid(predictions[:, :, :, -1])

        
        tp = tf.reduce_sum(pred_sigmoid * target_tensor)

        fp = tf.reduce_sum(pred_sigmoid * (1.-target_tensor))
                    
        fn = tf.reduce_sum((1 - pred_sigmoid) * target_tensor)

        tn = tf.reduce_sum((1-pred_sigmoid) * (1-target_tensor))

        recall_neg = tn / (tn + fp + 1e-8)
        recall_pos = tp  / (tp + fn + 1e-8)

        recall_neg = recall_neg / (tf.math.count_nonzero(1-target_tensor, dtype=tf.float32) + 1e-8)
        recall_pos = recall_pos / (tf.math.count_nonzero(target_tensor, dtype=tf.float32) + 1e-8)

        return recall_neg, recall_pos

    def get_precision_recall(self, truth, predictions):
        mask_true = tf.cast(tf.greater_equal(truth[:, :, :, -1],0.5), tf.int8)
        mask_not_true = tf.cast(tf.less(truth[:, :, :, -1],0.5), tf.int8)
        mask_pred = tf.cast(tf.greater_equal(tf.math.sigmoid(predictions[:, :, :, -1]),0.5), tf.int8)
        mask_not_pred = tf.cast(tf.less(tf.math.sigmoid(predictions[:, :, :, -1]),0.5), tf.int8)

        masks_and = tf.cast(tf.bitwise.bitwise_and(mask_true, mask_pred), tf.float32)
        tp = tf.math.count_nonzero(masks_and, dtype=tf.float32)

        masks_and_2 = tf.cast(tf.bitwise.bitwise_and(mask_not_true, mask_pred), tf.float32)
        fp = tf.math.count_nonzero(masks_and_2, dtype=tf.float32)

        masks_and_3 = tf.cast(tf.bitwise.bitwise_and(mask_true, mask_not_pred), tf.float32)
        fn = tf.math.count_nonzero(masks_and_3, dtype=tf.float32)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        return precision, recall
  

    def macro_double_soft_f1(self, truth, predictions):
        """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
        Use probability values instead of binary predictions.
        This version uses the computation of soft-F1 for both positive and negative class for each label.
        
        Args:
            y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
            
        Returns:
            cost (scalar Tensor): value of the cost function for the batch
        """
        y = truth[:, :, :, -1]
        y_hat = tf.math.sigmoid(predictions[:, :, :, -1])
        y = tf.cast(y, tf.float32)
        y_hat = tf.cast(y_hat, tf.float32)
        tp = tf.reduce_sum(y_hat * y)
        fp = tf.reduce_sum(y_hat * (1 - y))
        fn = tf.reduce_sum((1 - y_hat) * y)
        tn = tf.reduce_sum((1 - y_hat) * (1 - y))
        soft_f1_class1 = tp / (tp + fn + fp + 1e-16)
        soft_f1_class0 = tn / (tn + fp + fn + 1e-16)
        
        macro_cost_1 = 1 - (tf.reduce_sum(soft_f1_class1) / (tf.reduce_sum(y) + 1e-8))
        macro_cost_0 = 1 - (tf.reduce_sum(soft_f1_class0) / (tf.reduce_sum(1 - y) + 1e-8))
       
        return macro_cost_0, macro_cost_1

    def get_accracy_diffs(self, truth2, predictions2):
        truth = truth2[:, :, :, :8]
        predictions = predictions2[:, :, :, :8]

        mins = np.array([-0.5, -0.5, 0, 0.7, 0.1, 0.1, -1.1, -1.1])
        maxs = np.array([0.5, 0.5, 1, 1.9, 0.75, 0.95, 1.1, 1.1])

        mins = np.expand_dims(mins, [0, 1, 2])
        maxs = np.expand_dims(maxs, [0, 1, 2])
        
        truth = (truth + 1) / 2
        truth = truth * (maxs - mins) + mins 
        predictions = (predictions + 1) / 2
        predictions = predictions * (maxs - mins) + mins 

        theta_pred = tf.math.atan2(predictions[:, :, :, 6:7], predictions[:, :, :, 7:8]) * 180 / np.pi
        theta_truth = tf.math.atan2(truth[:, :, :, 6:7], truth[:, :, :, 7:8]) * 180 / np.pi
        theta_diff = tf.abs(theta_pred-theta_truth)
        theta_diff = tf.where(tf.greater_equal(truth2[:, :, :, 8:9],0.5), theta_diff, tf.zeros_like(truth2[:, :, :, 8:9]))
        true_count_theta = tf.math.count_nonzero(truth2[:, :, :, -1], dtype=tf.float32)
        accuracy_theta = tf.reduce_sum(theta_diff) / (true_count_theta + 1e-8)
        return accuracy_theta

        