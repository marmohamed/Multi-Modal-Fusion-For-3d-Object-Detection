import numpy as np
import cv2
from data.data_utils.velodyne_points import *
from utils.utils import *
import math
import os
import tensorflow as tf
from data.data_utils.data_reader import *


# def iou_box(box, anchor):
#     # box: x, y, z, w, h, l, theta
#     # anchor: x, y, z, w, h, l 
#     # iou = intersection / union

#     additional_width = box[5] * abs(math.cos(box[6]))
#     new_length = box[5] * abs(math.sin(box[6]))
#     new_width = box[3] + additional_width

#     new_box = box[:6]

#     if math.cos(box[6]) < 0:
#         new_box[0] = new_box[0] - additional_width
    
#     new_box[3] = new_width
#     new_box[5] = new_length

#     new_box[3:6] = new_box[:3] + new_box[3:6]
#     anchor[3:6] = anchor[:3] + anchor[3:6]

#     max_dim = np.max([new_box[:3], anchor[:3]], axis=0)
#     min_dim = np.min([new_box[3:6], anchor[3:6]], axis=0)

#     intersection_dim = min_dim - max_dim

#     intersection = intersection_dim[0] * intersection_dim[1] * intersection_dim[2]
#     vol_anchor = anchor[3] * anchor[4] * anchor[5]
#     vol_box = new_box[3] * new_box[4] * new_box[5] - (additional_width * new_length * box[4])
#     iou = intersection / (vol_box + vol_anchor - intersection)

#     return iou

# def sigmoid(x):
#   return 1 / (1 + math.exp(-x))


# np.array([8.04520593, 2.62727732, 9.24066019])
# def get_target(labels, anchors=np.array([[0, 1.63,1.5,3.89 ], [ 0, 2.08,2.59,6.05]]), input_size=(512, 448), output_size=(128, 112)):
# def get_target(labels, directions, anchors=np.array([3.9, 1.6, 1.5]), input_size=(512, 448), output_size=(128, 112)):
#     # ASSUMPTION: I will assume that the anchors contain a record for the height
#     """
#     - calculate the ratio = input size / final output size
#     - divide each of the x, y and z by the ratio
#     - subtract each of the x, y and z ....
#     """

#     ratio = input_size[0] // output_size[0]
#     y_target = np.zeros((output_size[0], output_size[1], 2, 9), np.float32)
   
#     for i in range(len(labels)):
#         label_i = np.array(labels[i])

#         x = int(label_i[0]/ratio)
#         y = int(label_i[1]/ratio)

#         if x >= output_size[0]:
#             x = output_size[0] - 1
#         if y >= output_size[1]:
#             y = output_size[1] - 1

#         if x < 0 or y < 0:
#             continue

#         label_i[0:2] = label_i[0:2] / (ratio*1.0)
#         label_i[2] = label_i[2] / 40.

#         if label_i[6] < 0:
#             label_i[6] += 3.14

#         angle = label_i[6] * 57.2958
#         if (angle >= 0 and angle <= 45) or (angle >= 135 and angle <= 225) or (angle >= 315 and angle <= 360):
#             k = 0
#         else:
#             k = 1

#         if label_i[6] >= 3 * np.pi / 4:
#             label_i[6] = label_i[6] - np.pi
        
#         label_i[6] = label_i[6] - k * (np.pi/2)

#         anchor = np.array([x+0.5, y+0.5, 1., anchors[0], anchors[1], anchors[2]])
    
#         label_i[:3] = (label_i[:3] - anchor[:3]) / anchor[3:6]
#         label_i[3:6] = np.log(label_i[3:6]/anchors)

#         y_target[x, y, k, :7] = label_i
#         y_target[x, y, k, 7:8] = [directions[i]]
#         y_target[x, y, k, 8:9] = [1]
        
#     return y_target



def get_target(labels, directions, anchors=np.array([3.9, 1.6, 1.5]), input_size=(800, 700), output_size=(200, 175)):
    # ASSUMPTION: I will assume that the anchors contain a record for the height
    """
    - calculate the ratio = input size / final output size
    - divide each of the x, y and z by the ratio
    - subtract each of the x, y and z ....
    """

    ratio = input_size[0] // output_size[0]
    y_target = np.zeros((output_size[0], output_size[1], 2, 9), np.float32)
    for i in range(len(labels)):
        label_i = np.array(labels[i])

        x = int(label_i[0]/ratio)
        y = int(label_i[1]/ratio)

        if x >= output_size[0]:
            x = output_size[0] - 1
        if y >= output_size[1]:
            y = output_size[1] - 1

        if x < 0 or y < 0:
            continue

        label_i[0:2] = label_i[0:2] / (ratio*1.0)
        label_i[2] = label_i[2] / 36.

        temp = label_i[6]
        angle = temp * 57.2958
        if (angle >= 0 and angle <= 45) or (angle <= 0 and angle >= -45)\
            or (angle >= 135 and angle <= 180) or (angle <= -135 and angle >= -180):
            if angle >= 135:
                label_i[6] = label_i[6] - np.pi
            elif angle <= -135:
                label_i[6] = label_i[6] + np.pi
            k = 0
        else:
            if angle < 0:
                label_i[6] = label_i[6] + np.pi
            k = 1

        label_i[6] = label_i[6] - k * (np.pi/2)
        
        anchor = np.array([x+0.5, y+0.5, 1., anchors[0], anchors[1], anchors[2]])
    
        label_i[:2] = (label_i[:2] - anchor[:2]) + 0.5
        label_i[3:6] = label_i[3:6]/(3*anchors)

        y_target[x, y, k, :7] = label_i
        y_target[x, y, k, 7:8] = [directions[i]]
        y_target[x, y, k, 8:9] = [1]
        
    return y_target