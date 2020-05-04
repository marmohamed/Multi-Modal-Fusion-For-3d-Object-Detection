
import numpy as np
import tensorflow as tf
import math
import numpy.matlib as npm



def nms(label, scores, max_output_size=100, iou_threshold=0.1, sess=None):
    boxes = []

    for j in range(0, len(label)):

        w = label[j][3]
        h = label[j][4] 
        x = label[j][0]
        y = label[j][1]
        a = label[j][6]
        

        polygon = convert5Pointto8Point(y, x, w, h, -a*57.2958)
        xs = polygon[0::2]
        ys = polygon[1::2]
            
        boxes.append([xs[0], ys[0], xs[2], ys[2]])

    boxes = np.array(boxes)

    selected_indices = tf.image.non_max_suppression(
                boxes, scores, max_output_size=max_output_size, iou_threshold=iou_threshold)
    if sess is not None:
        selected_indices = sess.run(selected_indices)
    else:
        with tf.Session() as sess:
            selected_indices = sess.run(selected_indices)

    return selected_indices




def convert5Pointto8Point(cx_, cy_, w_, h_, a_):

    theta = math.radians(a_)
    bbox = npm.repmat([[cx_], [cy_]], 1, 5) + \
       np.matmul([[math.cos(theta), math.sin(theta)],
                  [-math.sin(theta), math.cos(theta)]],
                 [[-w_ / 2, w_/ 2, w_ / 2, -w_ / 2, w_ / 2 + 8],
                  [-h_ / 2, -h_ / 2, h_ / 2, h_ / 2, 0]])
    # add first point
    x1, y1 = bbox[0][0], bbox[1][0]
    # add second point
    x2, y2 = bbox[0][1], bbox[1][1]
    # add third point
    #x3, y3 = bbox[0][4], bbox[1][4]   
    # add forth point
    x3, y3 = bbox[0][2], bbox[1][2]
    # add fifth point
    x4, y4 = bbox[0][3], bbox[1][3]

    return [x1, y1, x2, y2, x3, y3, x4, y4]










