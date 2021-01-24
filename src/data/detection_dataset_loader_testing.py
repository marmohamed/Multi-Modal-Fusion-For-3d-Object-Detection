import numpy as np
import cv2
from data.data_utils.velodyne_points import *
from utils.utils import *
import math
import os
import random

from data.data_utils.data_reader import *
from data.data_utils.target_utils import *
from data.data_utils.fv_utils import *
from data.dataset_loader import *

from scipy.spatial.transform import Rotation as R


class DetectionDatasetLoaderTesting(DatasetLoader):

    def _defaults(self, **kwargs):
        defaults = {
            'image_size': (370, 1224),
            'lidar_size': (448, 512, 40), 
            'anchors': np.array([3.9, 1.6, 1.5])
        }
        for k in kwargs:
            if k in defaults:
                defaults[k] = kwargs[k]
        return defaults
        

    def _init_generator(self, random_split = False):
        

        # if random_split:

            list_files = list(map(lambda x: x.split('.')[0], os.listdir(self.base_path+'/data_object_image_3/testing/image_3')))

            self.list_camera_paths = list(map(lambda x: self.base_path+'/data_object_image_3/testing/image_3/' + x + '.png', list_files[:]))
            self.list_lidar_paths = list(map(lambda x: self.base_path+'/data_object_velodyne/testing/velodyne/' + x + '.bin', list_files[:]))
            # self.list_label_paths = list(map(lambda x: self.base_path + '/data_object_label_2/testing/label_2/' + x + '.txt', list_files[:]))
            self.list_calib_paths = list(map(lambda x: self.base_path + '/data_object_calib/testing/calib/' + x + '.txt', list_files[:]))

            return self.__data_generator(self.base_path, 
                                    image_size=self.params['image_size'],
                                    lidar_size=self.params['lidar_size'], 
                                    anchors=self.params['anchors'],
                                    list_camera_paths=self.list_camera_paths[:], 
                                    list_lidar_paths=self.list_lidar_paths[:], 
                                    # list_label_paths=self.list_label_paths[:], 
                                    list_calib_paths=self.list_calib_paths[:])
                    
    def reset_generator(self):

        self.generator = self.__data_generator(self.base_path, 
                                    image_size=self.params['image_size'],
                                    lidar_size=self.params['lidar_size'], 
                                    anchors=self.params['anchors'],
                                    list_camera_paths=self.list_camera_paths[:], 
                                    list_lidar_paths=self.list_lidar_paths[:], 
                                    # list_label_paths=self.list_label_paths[:], 
                                    list_calib_paths=self.list_calib_paths[:])
        

    def get_next(self, batch_size=1):
        camera_tensors = []
        lidar_tensors = []
        # label_tensors = []

        for _ in range(batch_size):
            camera_tensor, lidar_tensor, label_tensor = list(next(self.generator))
            camera_tensors.append(camera_tensor)
            lidar_tensors.append(lidar_tensor)
            # label_tensors.append(label_tensor)

        camera_tensors = np.array(camera_tensors)
        lidar_tensors = np.array(lidar_tensors)
        # label_tensors = np.array(label_tensors)

        return (camera_tensors, lidar_tensors)

    def get_augmentation_parameters(self):
 
                    image_translate_x = 0
                    image_translate_y = 0

                    translate_x = 0
                    translate_y = 0
                    translate_z = 0
                    ang = 0

                    r = R.from_rotvec(np.radians(0) * np.array([0, 0, 1]))
                    rot = r.as_dcm()
                    rot = np.append(rot, np.array([[0,0,0]]), axis=0)
                    rot = np.append(rot, np.array([[0],[0],[0],[1]]), axis=1)

                    tr_x = 0
                    tr_y = 0
                    tr_z = 0
                    tr = np.array([[tr_x], [tr_y], [tr_z], [0]])

                    sc_x = 1
                    sc_y = 1
                    sc_z = 1
                    sc = np.array([[sc_x, 0, 0, 0], [0, sc_y, 0, 0], [0, 0, sc_z, 0], [0, 0, 0, 1]])

                    fliplr = False

                    return rot, tr, sc, image_translate_x, image_translate_y, ang, fliplr



    def __data_generator(self, base_path, image_size, lidar_size, anchors, 
                        list_camera_paths, list_lidar_paths, list_calib_paths):

        

        for camera_path, lidar_path, calib_path in zip(list_camera_paths, list_lidar_paths, list_calib_paths):
                
                rot, tr, sc, image_translate_x, image_translate_y, ang, fliplr = self.get_augmentation_parameters()
                
                data_reader_obj = DataReader(camera_path, calib_path, None, lidar_path, rot, sc, tr, ang, image_translate_x, image_translate_y, fliplr=fliplr)

                camera_image = data_reader_obj.read_image()
                lidar_image = data_reader_obj.lidar_reader.read_lidar()
                 
                yield(camera_image, lidar_image, None)




           


