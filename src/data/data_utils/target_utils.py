import numpy as np
import cv2
from data.data_utils.velodyne_points import *
from utils.utils import *
import math
import os
import tensorflow as tf
from data.data_utils.data_reader import *


def get_target(labels, directions, anchors=np.array([3.9, 1.6, 1.5]), input_size=(448, 512), output_size=(112, 128, 35)):
    ratio = input_size[0] // output_size[0]
    ratio = 1
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

        # label_i[0:2] = label_i[0:2] / (ratio*1.0)
        label_i[2] = label_i[2] / (output_size[2]*1.)
        ang = label_i[6]

        if ang < 0:
          dir_ = 0
        else:
          dir_ = 1
        while ang < 0:
          ang += np.pi
        k = 0
        if (ang > np.pi/4 and ang < (3/4.)* np.pi) or (ang < -np.pi/4 and ang > -(3./4.)* np.pi):
          k = 1

        if ang > (3./4.) * np.pi:
          ang -= np.pi
        label_i[6] = ang - k * (np.pi/2)
        label_i = np.append(label_i, [dir_])
        # label_i[6:8] = [math.sin(ang), math.cos(ang)]
        
        anchor = np.array([x+0.5, y+0.5, 0.5, anchors[0], anchors[1], anchors[2]])
    
        label_i[:3] = (label_i[:3] - anchor[:3]) 
        label_i[3:6] = np.log(label_i[3:6])

        
        # mins = np.array([-0.5, -0.5, -0.5, 0.7, 0.1, 0.1, -1.1, -1.1])
        # maxs = np.array([0.5, 0.5, 0.5, 1.9, 0.75, 0.91, 1.1, 1.1])
        
        # label_i = ((label_i - mins) / (maxs-mins)) * 2 - 1

        y_target[x, y, k, :8] = label_i
        y_target[x, y, k, 8:9] = [1]
        
    return y_target