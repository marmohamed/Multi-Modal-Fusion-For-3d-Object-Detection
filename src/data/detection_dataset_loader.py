import numpy as np
import cv2
from data.data_utils.velodyne_points import *
from utils.utils import *
import math
import os
import random

from data.data_utils.reader import *
from data.data_utils.target_utils import *
from data.data_utils.fv_utils import *
from data.dataset_loader import *


class DetectionDatasetLoader(DatasetLoader):

    def _defaults(self, **kwargs):
        defaults = {
            'image_size': (370, 1224),
            'lidar_size': (512, 448, 40), 
            'anchors': np.array([3.9, 1.6, 1.5])
        }
        for k in kwargs:
            if k in defaults:
                defaults[k] = kwargs[k]
        return defaults
        

    def _init_generator(self):

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

        augment = self.training
        return self.__data_generator(self.base_path, 
                                    image_size=self.params['image_size'],
                                    lidar_size=self.params['lidar_size'], 
                                    anchors=self.params['anchors'],
                                    list_camera_paths=self.list_camera_paths[:], 
                                    list_lidar_paths=self.list_lidar_paths[:], 
                                    list_label_paths=self.list_label_paths[:], 
                                    list_calib_paths=self.list_calib_paths[:], 
                                    augment_translate=augment, 
                                    augment_rotate=augment,
                                    negative_samples=augment,
                                    training=self.training)
                    
    def reset_generator(self):
        augment = self.training
        self.generator = self.__data_generator(self.base_path, 
                                    image_size=self.params['image_size'],
                                    lidar_size=self.params['lidar_size'], 
                                    anchors=self.params['anchors'],
                                    list_camera_paths=self.list_camera_paths[:], 
                                    list_lidar_paths=self.list_lidar_paths[:], 
                                    list_label_paths=self.list_label_paths[:], 
                                    list_calib_paths=self.list_calib_paths[:], 
                                    augment_translate=augment, 
                                    augment_rotate=augment,
                                    negative_samples=augment,
                                    training=self.training)
        

    def get_next(self, batch_size=1):
        camera_tensors = []
        lidar_tensors = []
        fv_velo_tensors = []
        label_tensors = []
        Tr_velo_to_cams = []
        R0_rects = []
        P3s = [] 
        shift_hs = []
        shift_ws = [] 

        for _ in range(batch_size):
            camera_tensor, lidar_tensor, label_tensor, Tr_velo_to_cam, R0_rect, P3, shift_h, shift_w = list(next(self.generator))
            camera_tensors.append(camera_tensor)
            lidar_tensors.append(lidar_tensor)
            label_tensors.append(label_tensor)
            Tr_velo_to_cams.append(Tr_velo_to_cam)
            R0_rects.append(R0_rect)
            P3s.append(P3)
            shift_hs.append(shift_h)
            shift_ws.append(shift_w)

        camera_tensors = np.array(camera_tensors)
        lidar_tensors = np.array(lidar_tensors)
        label_tensors = np.array(label_tensors)
        
        Tr_velo_to_cams = np.array(Tr_velo_to_cams)
        R0_rects = np.array(R0_rects)
        P3s = np.array(P3s)
        shift_hs = np.array(shift_hs)
        shift_ws = np.array(shift_hs)
        return (camera_tensors, lidar_tensors, label_tensors, Tr_velo_to_cams, R0_rects, P3s, shift_hs, shift_ws)


    def __data_generator(self, base_path, image_size, lidar_size, anchors, list_camera_paths, list_lidar_paths, list_label_paths, list_calib_paths, 
                        augment_translate=False, augment_rotate=False, negative_samples=False, training=True):

        if training:
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
                camera_image, shift_h, shift_w = read_camera(camera_path, image_size)
                h, w, _ = cv2.imread(camera_path).shape
                lidar_image = read_lidar(lidar_path, calib_path, lidar_size, img_height=h, img_width=w)
                velo_front_view = read_pc_fv(calib_path, lidar_path)
                _, label, Tr_velo_to_cam, R0_rect, P3, directions = read_label(label_path, calib_path, shift_h, shift_w)
                label = get_target(label, directions,  anchors=anchors)
                camera_image = camera_image / 255.
                lidar_image = lidar_image / 255.
                yield(camera_image, lidar_image, label,
                            np.concatenate([np.array(Tr_velo_to_cam).reshape((3, 4)), np.array([[0, 0, 0, 1]])], axis=0),
                            np.concatenate([np.concatenate([np.array(R0_rect).reshape((3, 3)), np.array([[0], [0], [0]])], axis=1),  np.array([[0, 0, 0, 1]])], axis=0),
                            np.array(P3).reshape((3, 4)), 
                            np.array([shift_h]), 
                            np.array([shift_w])
                            )

                if augment_rotate:
                        translate_x = 0
                        translate_y = 0
                        translate_z = 0
                        ang = random.randint(-5, 5)
                    
                        camera_image, shift_h, shift_w = read_camera(camera_path, image_size)
                        h, w, _ = cv2.imread(camera_path).shape
                        lidar_image = read_lidar(lidar_path, calib_path, lidar_size, img_height=h, img_width=w, ang=ang, translate_x=translate_x, translate_y=translate_y)
                        _, label, Tr_velo_to_cam, R0_rect, P3, directions = read_label(label_path, calib_path, shift_h, shift_w, translate_x=translate_x, translate_y=translate_y, ang=ang)
                        label = get_target(label, directions,  anchors=anchors)
                        camera_image = camera_image / 255.
                        lidar_image = lidar_image / 255.
                        yield(camera_image, lidar_image, label,
                                    np.concatenate([np.array(Tr_velo_to_cam).reshape((3, 4)), np.array([[0, 0, 0, 1]])], axis=0),
                                    np.concatenate([np.concatenate([np.array(R0_rect).reshape((3, 3)), np.array([[0], [0], [0]])], axis=1),  np.array([[0, 0, 0, 1]])], axis=0),
                                    np.array(P3).reshape((3, 4)), 
                                    np.array([shift_h]), 
                                    np.array([shift_w])
                                    )

                if augment_translate:
                        translate_x = random.randint(-5, 5)
                        translate_y = random.randint(-5, 5)
                        translate_z = random.random() - 0.5
                        translate_z = 0
                        ang = 0
                        
                        camera_image, shift_h, shift_w = read_camera(camera_path, image_size)
                        h, w, _ = cv2.imread(camera_path).shape
                        lidar_image = read_lidar(lidar_path, calib_path, lidar_size, img_height=h, img_width=w, ang=ang, translate_x=translate_x, translate_y=translate_y, translate_z=translate_z)
                        _, label, Tr_velo_to_cam, R0_rect, P3, directions = read_label(label_path, calib_path, shift_h, shift_w, translate_x=translate_x, translate_y=translate_y, translate_z=-translate_z, ang=ang)
                        label = get_target(label, directions,  anchors=anchors)
                        camera_image = camera_image / 255.
                        lidar_image = lidar_image / 255.
                        yield(camera_image, lidar_image, label,
                                    np.concatenate([np.array(Tr_velo_to_cam).reshape((3, 4)), np.array([[0, 0, 0, 1]])], axis=0),
                                    np.concatenate([np.concatenate([np.array(R0_rect).reshape((3, 3)), np.array([[0], [0], [0]])], axis=1),  np.array([[0, 0, 0, 1]])], axis=0),
                                    np.array(P3).reshape((3, 4)), 
                                    np.array([shift_h]), 
                                    np.array([shift_w])
                                    )

                if augment_rotate and augment_translate:
                        translate_x = random.randint(-5, 5)
                        translate_y = random.randint(-5, 5)
                        translate_z = random.random() - 0.5
                        translate_z = 0
                        ang = random.randint(-5, 5)
                    
                        camera_image, shift_h, shift_w = read_camera(camera_path, image_size)
                        h, w, _ = cv2.imread(camera_path).shape
                        lidar_image = read_lidar(lidar_path, calib_path, lidar_size, img_height=h, img_width=w, ang=ang, translate_x=translate_x, translate_y=translate_y, translate_z=translate_z)
                        _, label, Tr_velo_to_cam, R0_rect, P3, directions = read_label(label_path, calib_path, shift_h, shift_w, translate_x=translate_x, translate_y=translate_y, translate_z=-translate_z, ang=ang)
                        label = get_target(label, directions,  anchors=anchors)
                        camera_image = camera_image / 255.
                        lidar_image = lidar_image / 255.
                        yield(camera_image, lidar_image, label,
                                    np.concatenate([np.array(Tr_velo_to_cam).reshape((3, 4)), np.array([[0, 0, 0, 1]])], axis=0),
                                    np.concatenate([np.concatenate([np.array(R0_rect).reshape((3, 3)), np.array([[0], [0], [0]])], axis=1),  np.array([[0, 0, 0, 1]])], axis=0),
                                    np.array(P3).reshape((3, 4)), 
                                    np.array([shift_h]), 
                                    np.array([shift_w])
                                    )


                        translate_x = random.randint(-5, 5)
                        translate_y = random.randint(-5, 5)
                        translate_z = random.random() - 0.5
                        translate_z = 0
                        ang = random.randint(-15, -5)
                    
                        camera_image, shift_h, shift_w = read_camera(camera_path, image_size)
                        h, w, _ = cv2.imread(camera_path).shape
                        lidar_image = read_lidar(lidar_path, calib_path, lidar_size, img_height=h, img_width=w, ang=ang, translate_x=translate_x, translate_y=translate_y, translate_z=translate_z)
                        _, label, Tr_velo_to_cam, R0_rect, P3, directions = read_label(label_path, calib_path, shift_h, shift_w, translate_x=translate_x, translate_y=translate_y, translate_z=-translate_z, ang=ang)
                        label = get_target(label, directions,  anchors=anchors)
                        camera_image = camera_image / 255.
                        lidar_image = lidar_image / 255.
                        yield(camera_image, lidar_image, label,
                                    np.concatenate([np.array(Tr_velo_to_cam).reshape((3, 4)), np.array([[0, 0, 0, 1]])], axis=0),
                                    np.concatenate([np.concatenate([np.array(R0_rect).reshape((3, 3)), np.array([[0], [0], [0]])], axis=1),  np.array([[0, 0, 0, 1]])], axis=0),
                                    np.array(P3).reshape((3, 4)), 
                                    np.array([shift_h]), 
                                    np.array([shift_w])
                                    )


                        translate_x = random.randint(-5, 5)
                        translate_y = random.randint(-5, 5)
                        translate_z = random.random() - 0.5
                        translate_z = 0
                        ang = random.randint(5, 15)
                    
                        camera_image, shift_h, shift_w = read_camera(camera_path, image_size)
                        h, w, _ = cv2.imread(camera_path).shape
                        lidar_image = read_lidar(lidar_path, calib_path, lidar_size, img_height=h, img_width=w, ang=ang, translate_x=translate_x, translate_y=translate_y, translate_z=translate_z)
                        _, label, Tr_velo_to_cam, R0_rect, P3, directions = read_label(label_path, calib_path, shift_h, shift_w, translate_x=translate_x, translate_y=translate_y, translate_z=-translate_z, ang=ang)
                        label = get_target(label, directions,  anchors=anchors)
                        camera_image = camera_image / 255.
                        lidar_image = lidar_image / 255.
                        yield(camera_image, lidar_image, label,
                                    np.concatenate([np.array(Tr_velo_to_cam).reshape((3, 4)), np.array([[0, 0, 0, 1]])], axis=0),
                                    np.concatenate([np.concatenate([np.array(R0_rect).reshape((3, 3)), np.array([[0], [0], [0]])], axis=1),  np.array([[0, 0, 0, 1]])], axis=0),
                                    np.array(P3).reshape((3, 4)), 
                                    np.array([shift_h]), 
                                    np.array([shift_w])
                                    )


                        


