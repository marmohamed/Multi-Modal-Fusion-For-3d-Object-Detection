import tensorflow as tf
import numpy as np

def iou_box_gpu(box, anchor):
    # box: x, y, z, w, h, l, theta
    # anchor: x, y, z, w, h, l 
    # iou = intersection / union

    additional_width = tf.multiply(box[:, 5:6], tf.abs(tf.cos(box[:, 6:7])))
    new_length = tf.multiply(box[:, 5:6], tf.abs(tf.sin(box[:, 6:7])))
    new_width = tf.add(box[:, 3:4], additional_width)

    new_box = box[:, :6]
    temp_00 = tf.where(tf.less_equal(tf.cos(box[:, 6:7]), 0), additional_width, tf.zeros_like(additional_width))
    temp_0 =  new_box[:, 0:1] - temp_00
    new_box = tf.concat([temp_0, box[:, 1:3], new_width, box[:, 4:5], new_length, box[:, 5:6]], axis=1)
    temp_3_6 = new_box[:, :3] + new_box[:, 3:6]
    new_box = tf.concat([new_box[:, :3], temp_3_6], axis=1)
    temp2_3_6 = anchor[:, :3] + anchor[:, 3:6]
    anchor = tf.concat([anchor[:, :3], temp2_3_6, anchor[:, 6:]], axis=1)

    max_dim = tf.maximum(new_box[:, :3], anchor[:, :3])
    min_dim = tf.minimum(new_box[:, 3:6], anchor[:, 3:6])

    intersection_dim = min_dim - max_dim

    intersection = intersection_dim[:, 0:1] * intersection_dim[:, 1:2] * intersection_dim[:, 2:3]
    vol_anchor = anchor[:, 3:4] * anchor[:, 4:5] * anchor[:, 5:6]
    vol_box = new_box[:, 3:4] * new_box[:, 4:5] * new_box[:, 5:6] - (additional_width * new_length * box[:, 4:5])
    iou = intersection / (vol_box + vol_anchor - intersection)
    return iou

def nms(preds, threshold):

    def nms_2_boxes(box1, box2):
        temp = tf.logical_and(
                tf.greater_equal(box1[:, 0, 7:8], 0.5), 
                tf.greater_equal(box2[:, 0, 7:8], 0.5))
        iou = tf.where(temp, iou_box_gpu(box1[:, 0, :], box2[:, 0, :]), tf.zeros_like(temp, dtype=tf.float32))
        prob = tf.where(tf.greater_equal(iou, threshold), tf.ones_like(iou, dtype=tf.bool), tf.zeros_like(iou, dtype=tf.bool))
        prob0 = tf.where(tf.logical_and(prob, 
                                        tf.greater_equal(box1[:, 0, 7:8], box2[:, 0, 7:8])),
                                        box1[:, 0, 7:8], tf.zeros_like(box1[:, 0, 7:8]))
        prob1 = tf.where(tf.logical_and(prob,
                                        tf.greater_equal(box1[:, 0, 7:8], box1[:, 0, 7:8])),
                                        box2[:, 0, 7:8], tf.zeros_like(box2[:, 0, 7:8]))

        prob0 = box1[:, 0:1, 7:8]*prob0
        prob1 = box2[:, 0:1, 7:8]*prob1

        prob0 =  tf.concat([box1[:, :, :7], prob0, box1[:, :, 8:]], axis=2)
        prob1 = tf.concat([box2[:, :, :7], prob1, box2[:, :, 8:]], axis=2)
        
        return prob0, prob1

    def fn_shape_2(i, j):
        i = tf.cast(i, tf.int32)
        j = tf.cast(j, tf.int32)
        boxes = preds[:, i, j, :, :]

        prob0, prob1 = nms_2_boxes(boxes[:, 0:1, :], boxes[:, 1:2, :])

        boxes = tf.concat([prob0, prob1], axis=1)

        def fn1(x, y):
            other_boxes = preds[:, x, y, :, :]

            prob0, prob1 = nms_2_boxes(boxes[:, 0:1, :], other_boxes[:, 0:1, :])
            prob0, prob1 = nms_2_boxes(prob0, other_boxes[:, 1:2, :])

            prob0_1, prob1 = nms_2_boxes(boxes[:, 1:2, :], other_boxes[:, 0:1, :])
            prob0_1, prob1 = nms_2_boxes(prob0_1, other_boxes[:, 1:2, :])

            return tf.concat([prob0, prob0_1], axis=1)

        def fn2():
            return boxes

        boxes = tf.cond(i + 1 < preds.get_shape()[1], lambda: fn1(i+1, j), lambda: fn2())
        boxes = tf.cond(j + 1 < preds.get_shape()[2], lambda: fn1(i, j + 1), lambda: fn2())
        boxes = tf.cond(i - 1 >= 0, lambda: fn1(i-1, j), lambda: fn2())
        boxes = tf.cond(j - 1 >= 0, lambda: fn1(i, j-1), lambda: fn2())

        boxes = tf.cond(tf.logical_and(i + 1 < preds.get_shape()[1], j + 1 < preds.get_shape()[2]), lambda: fn1(i+1, j+1), lambda: fn2())
        boxes = tf.cond(tf.logical_and(i + 1 < preds.get_shape()[1], j - 1 >= 0), lambda: fn1(i + 1, j - 1), lambda: fn2())
        boxes = tf.cond(tf.logical_and(i - 1 >= 0, j + 1 < preds.get_shape()[2]), lambda: fn1(i - 1, j + 1), lambda: fn2())
        boxes = tf.cond(tf.logical_and(i - 1 >= 0, j + 1 < preds.get_shape()[2]), lambda: fn1(i - 1, j + 1), lambda: fn2())

        return boxes

    def fn_shape_1(i):
        return tf.map_fn(lambda j: fn_shape_2(i, j), tf.constant(list(range(preds.get_shape()[2])), dtype=tf.float32))

    preds = tf.map_fn(lambda i: fn_shape_1(i), tf.constant(list(range(preds.get_shape()[1])), dtype=tf.float32))

    preds = tf.transpose(preds, [2, 0, 1, 3, 4])

    return preds

