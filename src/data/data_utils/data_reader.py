import numpy as np
import cv2
from data.data_utils.velodyne_points import *
from data.data_utils.calib_reader import *
from data.data_utils.image_reader import *
from data.data_utils.label_reader import *
import math
from utils.utils import *
import os
import tensorflow as tf
import matplotlib.patches as patches


from data.data_utils.fv_utils import *

class DataReader:

    def __init__(self, image_path, calib_path, label_path, lidar_path, 
                    rot, sc, tr, ang, 
                    translate_x, translate_y, 
                    get_actual_dims=False, 
                    from_file=True):
        self.image_path = image_path
        self.calib_path = calib_path
        self.label_path = label_path
        self.lidar_path = lidar_path
        self.rot = rot
        self.sc = sc
        self.tr = tr
        self.lidar_reader = LidarReader(lidar_path, calib_path, image_path, rot, tr, sc)
        self.image_reader = ImageReader(image_path, translate_x, translate_y)
        self.calib_reader = CalibReader(calib_path)
        self.label_reader = LabelReader(label_path, calib_path, rot, tr, sc, ang, self.calib_reader, 
                                        get_actual_dims=get_actual_dims, 
                                        from_file=from_file)

 
    def read_lidar(self):
        return self.lidar_reader.read_lidar()

    def read_calib(self):
        return self.calib_reader.read_calib()

    def read_image(self):
        return self.image_reader.read_image()

    def read_label(self):
        return self.label_reader.read_label()
    