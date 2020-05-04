import tensorflow as tf
import numpy as np

def prepare_anchors():
    anchors = np.zeros((128, 112, 2, 6), dtype=float)
    ratio = 4
    for i in range(128):
        for j in range(112):
            anchors[i, j, :, :] = [[i+0.5, j+0.5, 0.5, 3.9, 1.6, 1.5] , 
                                   [i+0.5, j+0.5, 0.5, 3.9, 1.6, 1.5]]
            # anchors[i, j, :, :] = [[i*ratio, j*ratio, 1, 1.63,1.5,3.89 ], [ i*ratio, j*ratio, 1, 2.08,2.59,6.05]]
    # anchors = tf.constant(anchors, dtype=tf.float32)
    anchors = np.expand_dims(anchors, 0)
    return anchors

def adjust_predictions(model_output, anchors, epsilon=1e-10):
    # x,y,z,w,h,d,t , pc, lc
    # (?, 128, 112, 2, 9)
    # x,y,z,w,h,d,t
    temp1 = tf.div(tf.subtract(model_output[:, :, :, :, :3], anchors[:, :, :, :, :3]), anchors[:, :, :, :, 3:6]) 
                
    # temp1 = anchors[:, :, :, :, :3]
        
    div_value = tf.div(model_output[:, :, :, :, 3:6], anchors[:, :, :, :, 3:6])
    temp2 = tf.where(tf.greater(anchors[:, :, :, :, 3:6], 0.0), 
                tf.log(tf.clip_by_value(div_value, epsilon, div_value)), 
                tf.zeros_like(anchors[:, :, :, :, 3:6]))
    
    # TODO orienation
    # temp3 = np.array([0, np.pi/2.]) - model_output[:, :, :, :, 6]
    temp3 = model_output[:, :, :, :, 6]
    temp3 = tf.expand_dims(temp3, 4)
    model_output = tf.concat([temp1, temp2, temp3, model_output[:, :, :, :, 7:]], axis=4)
    return model_output



