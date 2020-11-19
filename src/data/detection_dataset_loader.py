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


class DetectionDatasetLoader(DatasetLoader):

    def _defaults(self, **kwargs):
        defaults = {
            'image_size': (370, 1224),
            'lidar_size': (448, 512, 35), 
            'anchors': np.array([3.9, 1.6, 1.5])
        }
        for k in kwargs:
            if k in defaults:
                defaults[k] = kwargs[k]
        return defaults
        

    def _init_generator(self, random_split = False):
        

        if random_split:

            list_files = list(map(lambda x: x.split('.')[0], os.listdir(self.base_path+'/data_object_image_3/training/image_3')))
            random.seed(self.random_seed)
            random.shuffle(list_files)

            camera_paths = list(map(lambda x: self.base_path+'/data_object_image_3/training/image_3/' + x + '.png', list_files))
            lidar_paths = list(map(lambda x: self.base_path+'/data_object_velodyne/training/velodyne/' + x + '.bin', list_files))
            label_paths = list(map(lambda x: self.base_path + '/data_object_label_2/training/label_2/' + x + '.txt', list_files))
            calib_paths = list(map(lambda x: self.base_path + '/data_object_calib/training/calib/' + x + '.txt', list_files))
            
            if self.num_samples is None:
                ln = int(len(list_files) * self.training_per)
                final_sample = len(list_files)
            else:
                ln = int(self.num_samples * self.training_per)
                final_sample = self.num_samples

            if self.training:
                self.list_camera_paths = camera_paths[:ln]
                self.list_lidar_paths = lidar_paths[:ln]
                self.list_label_paths = label_paths[:ln]
                self.list_calib_paths = calib_paths[:ln]
            else:
                self.list_camera_paths = camera_paths[ln:final_sample]
                self.list_lidar_paths = lidar_paths[ln:final_sample]
                self.list_label_paths = label_paths[ln:final_sample]
                self.list_calib_paths = calib_paths[ln:final_sample]
        else:
            if self.training:
                file_name = '/trainsplit.txt'
            else:
                file_name = '/valsplit.txt'
            with open(self.base_path + file_name, 'r') as f:
                list_file_nums = f.readlines()
            list_files = ['0'*(6-len(l.strip())) + l.strip() for l in list_file_nums]

            if self.num_samples is None:
                ln = int(len(list_files))
            else:
                ln = int(self.num_samples)


            self.list_camera_paths = list(map(lambda x: self.base_path+'/data_object_image_3/training/image_3/' + x + '.png', list_files[:ln]))
            self.list_lidar_paths = list(map(lambda x: self.base_path+'/data_object_velodyne/training/velodyne/' + x + '.bin', list_files[:ln]))
            self.list_label_paths = list(map(lambda x: self.base_path + '/data_object_label_2/training/label_2/' + x + '.txt', list_files[:ln]))
            self.list_calib_paths = list(map(lambda x: self.base_path + '/data_object_calib/training/calib/' + x + '.txt', list_files[:ln]))

        return self.__data_generator(self.base_path, 
                                    image_size=self.params['image_size'],
                                    lidar_size=self.params['lidar_size'], 
                                    anchors=self.params['anchors'],
                                    list_camera_paths=self.list_camera_paths[:], 
                                    list_lidar_paths=self.list_lidar_paths[:], 
                                    list_label_paths=self.list_label_paths[:], 
                                    list_calib_paths=self.list_calib_paths[:], 
                                    training=self.training)
                    
    def reset_generator(self):

        self.generator = self.__data_generator(self.base_path, 
                                    image_size=self.params['image_size'],
                                    lidar_size=self.params['lidar_size'], 
                                    anchors=self.params['anchors'],
                                    list_camera_paths=self.list_camera_paths[:], 
                                    list_lidar_paths=self.list_lidar_paths[:], 
                                    list_label_paths=self.list_label_paths[:], 
                                    list_calib_paths=self.list_calib_paths[:], 
                                    training=self.training)
        

    def get_next(self, batch_size=1):
        camera_tensors = []
        lidar_tensors = []
        label_tensors = []

        for _ in range(batch_size):
            camera_tensor, lidar_tensor, label_tensor = list(next(self.generator))
            camera_tensors.append(camera_tensor)
            lidar_tensors.append(lidar_tensor)
            label_tensors.append(label_tensor)

        camera_tensors = np.array(camera_tensors)
        lidar_tensors = np.array(lidar_tensors)
        label_tensors = np.array(label_tensors)

        return (camera_tensors, lidar_tensors, label_tensors)


    def apply_mask(self, image, size=30, n_squares=3):
        h, w, channels = image.shape
        new_image = image[:]
        for _ in range(n_squares):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - size // 2, 0, h)
            y2 = np.clip(y + size // 2, 0, h)
            x1 = np.clip(x - size // 2, 0, w)
            x2 = np.clip(x + size // 2, 0, w)
            new_image[y1:y2,x1:x2,:] = 0
        return new_image


    def apply_mask_lidar(self, image, size_x=5, size_y=5, size_z=5, n_squares=5):
        w, l, h = image.shape
        new_image = image[:]
        for _ in range(n_squares):
            x = np.random.randint(w)
            y = np.random.randint(l)
            z = np.random.randint(h)

            y1 = np.clip(y - size_y // 2, 0, l)
            y2 = np.clip(y + size_y // 2, 0, l)
            x1 = np.clip(x - size_x // 2, 0, w)
            x2 = np.clip(x + size_x // 2, 0, w)
            z1 = np.clip(z - size_z // 2, 0, h)
            z2 = np.clip(z + size_z // 2, 0, h)
            new_image[x1:x2, y1:y2,z1:z2] = 0.
        return new_image


    def get_augmentation_parameters(self):
        if self.augment:

                    if np.random.random_sample() >= 0.0:
                        image_translate_x = random.randint(-50, 50)
                    else:
                        image_translate_x = 0
                    if np.random.random_sample() >= 0.0:
                        image_translate_y = random.randint(-25, 25)
                    else:
                        image_translate_y = 0

                    if np.random.random_sample() >= 0.0:
                        translate_x = random.randint(-20, 20)
                    else:
                        translate_x = 0
                    if np.random.random_sample() >= 0.0:
                        translate_y = random.randint(-20, 20)
                    else:
                        translate_y = 0

                    if np.random.random_sample() >= 0.0:
                        translate_z = random.random() - 0.5
                    else:
                        translate_z = 0

                    if np.random.random_sample() >= 0.0:
                        ang = random.randint(-60, 60)
                    else:
                        ang = 0

                    r = R.from_rotvec(np.radians(ang) * np.array([0, 0, 1]))
                    rot = r.as_dcm()
                    rot = np.append(rot, np.array([[0,0,0]]), axis=0)
                    rot = np.append(rot, np.array([[0],[0],[0],[1]]), axis=1)

                    tr_x = translate_x
                    tr_y = translate_y
                    tr_z = translate_z
                    tr = np.array([[tr_x], [tr_y], [tr_z], [0]])
                    
                    translate_x = 0
                    translate_y = 0
                    translate_z = 0
                    
                    sc_x = 1
                    sc_y = 1
                    sc_z = 1

                    if np.random.random_sample() >= 0.0:
                       sc_x += ((random.random() * 2) - 1.) / 5.

                    if np.random.random_sample() >= 0.0:
                       sc_y += ((random.random() * 2) - 1.) / 5.


                    sc = np.array([[sc_x, 0, 0, 0], [0, sc_y, 0, 0], [0, 0, sc_z, 0], [0, 0, 0, 1]])

                    fliplr = np.random.random_sample() >= 0.5

        else:
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
                        list_camera_paths, list_lidar_paths, list_label_paths, list_calib_paths, 
                        training=True):

        if training and self.augment:
            value = random.randint(0, 50)
            random.seed(value)
            random.shuffle(list_camera_paths)
            random.seed(value)
            random.shuffle(list_lidar_paths)
            random.seed(value)
            random.shuffle(list_label_paths)
            random.seed(value)
            random.shuffle(list_calib_paths)

        for camera_path, lidar_path, label_path, calib_path in zip(list_camera_paths, list_lidar_paths, list_label_paths, list_calib_paths):
                
                rot, tr, sc, image_translate_x, image_translate_y, ang, fliplr = self.get_augmentation_parameters()
                
                data_reader_obj = DataReader(camera_path, calib_path, label_path, lidar_path, rot, sc, tr, ang, image_translate_x, image_translate_y, fliplr=fliplr)

                camera_image = data_reader_obj.read_image()
                lidar_image = data_reader_obj.lidar_reader.read_lidar()

                if self.augment:
                    if np.random.random_sample() >= 0.4:
                        noise = np.random.rand(448, 512, 35)
                        noise2 = np.random.rand(448, 512, 35)

                        noise = np.array(noise>=0.99, dtype=np.int)
                        noise2 = np.array(noise2>=0.99, dtype=np.int)

                        lidar_image = np.array(np.clip(lidar_image + noise*noise2, 0, 1), dtype=np.float)

                    if np.random.random_sample() >= 0.4:
                        noise = np.random.rand(448, 512, 35)
                        noise2 = np.random.rand(448, 512, 35)

                        noise = np.array(noise>=0.1, dtype=np.int)
                        noise2 = np.array(noise2>=0.1, dtype=np.int)

                        lidar_image = np.array(np.clip(lidar_image * noise*noise2, 0, 1), dtype=np.float)

                    if np.random.random_sample() >= 0.4:
                        lidar_image = self.apply_mask_lidar(lidar_image)

                _, label, directions = data_reader_obj.label_reader.read_label()
                label = get_target(label, directions,  anchors=anchors)
                 
                yield(camera_image, lidar_image, label)




           


