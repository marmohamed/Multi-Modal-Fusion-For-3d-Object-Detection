import numpy as np
import cv2
from utils.utils import *
import math
import os
import random
import matplotlib.image as mpimg
import json
from skimage.draw import polygon
from abc import ABC, abstractmethod, ABCMeta

from data.data_utils.data_reader import *
from data.data_utils.target_utils import *
from data.dataset_loader import *

class SegmentationDatasetLoader(DatasetLoader):

    __metaclass__ = ABCMeta

    def _defaults(self, **kwargs):
        defaults = {
            'image_size': (370, 1224),
            'out_size': (24, 78, 1)
        }
        for k in kwargs:
            if k in defaults:
                defaults[k] = kwargs[k]
        return defaults

    def reset_generator(self):
        self.generator = self._data_generator(self.base_path, 
                                    image_size=self.params['image_size'],
                                    out_size=self.params['out_size'], 
                                    list_camera_paths=self.list_camera_paths, 
                                    list_label_paths=self.list_label_paths)


    def get_next(self, batch_size=1):
        images = []
        labels_cars = []
        labels_ground = []
        for _ in range(batch_size):
            image, label_cars, label_ground = list(next(self.generator))
            images.extend(image)
            labels_cars.extend(label_cars)
            labels_ground.extend(label_ground)

        images = np.array(images)
        labels_cars = np.array(labels_cars)
        labels_ground = np.array(labels_ground)

        labels = np.concatenate([labels_cars, labels_ground], axis=-1)

        return images, labels

    @abstractmethod
    def _init_generator(self):
        pass

    @abstractmethod
    def _data_generator(self, base_path, image_size, out_size, list_camera_paths, list_label_paths):
        pass
    

class KITTISegmentationDatasetLoader(SegmentationDatasetLoader):
    
    def _init_generator(self):
        list_files = list(map(lambda x: x.split('.')[0], os.listdir(self.base_path+'/data_semantics/training/semantic_rgb/')))
        random.seed(self.random_seed)
        random.shuffle(list_files)

        camera_paths = list(map(lambda x: self.base_path+'/data_semantics/training/image_2/' + x + '.png', list_files))

        label_paths = list(map(lambda x: self.base_path + '/data_semantics/training/semantic_rgb/' + x + '.png', list_files))
        
        if self.num_samples is None:
            ln = int(len(list_files) * self.training_per)
            final_sample = len(list_files)
        else:
            ln = int(self.num_samples * self.training_per)
            final_sample = self.num_samples
        
        if self.training:
            self.list_camera_paths = camera_paths[:ln]
            self.list_label_paths = label_paths[:ln]
        else:
            self.list_camera_paths = camera_paths[ln:final_sample]
            self.list_label_paths = label_paths[ln:final_sample]

        return self._data_generator(self.base_path, 
                                    image_size=self.params['image_size'],
                                    out_size=self.params['out_size'], 
                                    list_camera_paths=self.list_camera_paths, 
                                    list_label_paths=self.list_label_paths)


    def _data_generator(self, base_path, image_size, out_size, list_camera_paths, list_label_paths):
        
        for camera_path, label_path in zip(list_camera_paths, list_label_paths):

            image = cv2.imread(camera_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            w, h, _ = image.shape
            image = cv2.resize(image, (1224, 370))

            label_img = mpimg.imread(label_path)

            label_img_cars = (label_img[:, :, 0] == 0) & (label_img[:, :, 1] == 0) & (label_img[:, :, 2] >= 0.5)
            label_img_cars = cv2.resize(label_img_cars.astype(np.float32), (out_size[1], out_size[0]))
            label_img_cars_black = 1 - (label_img_cars < 0.5)
            label_img_cars_blue = label_img_cars>= 0.5
            label_img_cars = (label_img_cars_black + label_img_cars_blue)/2
            label_img_cars = label_img_cars.reshape(out_size).astype(np.float32)

            label_img_ground = (label_img[:, :, 0] >= 0.5) & (label_img[:, :, 0] <= 0.55) & (label_img[:, :, 1] >= 0.25) &\
                                 (label_img[:, :, 1] <= 0.30) & (label_img[:, :, 2] >= 0.5) & (label_img[:, :, 2] <= 0.55)   
            label_img_ground = cv2.resize(label_img_ground.astype(np.float32), (out_size[1], out_size[0]))
            label_img_ground_black = 1 - (label_img_ground < 0.5)
            label_img_ground_blue = label_img_ground>= 0.5
            label_img_ground = (label_img_ground_black + label_img_ground_blue)/2
            label_img_ground = label_img_ground.reshape(out_size).astype(np.float32)
            
            images = [image/255., image[:, ::-1, :]/255.]
            labels = [label_img_cars, label_img_cars[:, ::-1, :]]
            labels_ground = [label_img_ground, label_img_ground[:, ::-1, :]]
            
            temp_image = image.copy()
            temp_image[:, -100:, :] = 0
            temp_image[:, :100, :] = 0
            images.append(temp_image/255.)

            temp_label = label_img_cars.copy()
            temp_label[:, -int((100/370)*out_size[0]):, :] = 0
            temp_label[:, :int((100/370)*out_size[0]), :] = 0
            labels.append(temp_label)

            temp_label = label_img_ground.copy()
            temp_label[:, -int((100/370)*out_size[0]):, :] = 0
            temp_label[:, :int((100/370)*out_size[0]), :] = 0
            labels_ground.append(temp_label)

            temp_image = image.copy()
            temp_image[-50:, :, :] = 0
            temp_image[:50, :, :] = 0
            images.append(temp_image/255.)

            temp_label = label_img_cars.copy()
            temp_label[-int((50/1224)*out_size[1]):, :, :] = 0
            temp_label[:int((50/1224)*out_size[1]), :, :] = 0
            labels.append(temp_label)

            temp_label = label_img_ground.copy()
            temp_label[-int((50/1224)*out_size[1]):, :, :] = 0
            temp_label[:int((50/1224)*out_size[1]), :, :] = 0
            labels_ground.append(temp_label)

            yield (images, labels, labels_ground)

        


class CityScapesSegmentationDatasetLoader(SegmentationDatasetLoader):


    def _init_generator(self):
        dir_name = 'train'
        if not self.training:
            dir_name = 'val'
        cities = os.listdir(self.base_path+'/leftImg8bit_trainvaltest/leftImg8bit/' + dir_name + '/')
        list_files = []
        for city in cities:
            tmp = os.listdir(self.base_path+'/leftImg8bit_trainvaltest/leftImg8bit/' + dir_name + '/' + city)
            tmp = list(map(lambda x: city + '/' + '_'.join(x.split('_')[:3]), tmp))
            list_files.extend(tmp)
        
        random.seed(self.random_seed)
        random.shuffle(list_files)

        camera_paths = list(map(lambda x: self.base_path+'/leftImg8bit_trainvaltest/leftImg8bit/' + dir_name + '/' + x + '_leftImg8bit.png', list_files))

        label_paths = list(map(lambda x: self.base_path + '/gtFine_trainvaltest/gtFine/' + dir_name + '/' + x + '_gtFine_polygons.json', list_files))
        
        if self.num_samples is None:
            ln = int(len(list_files) * self.training_per)
            final_sample = len(list_files)
        else:
            ln = int(self.num_samples * self.training_per)
            final_sample = self.num_samples
        
        if self.training:
            self.list_camera_paths = camera_paths[:ln]
            self.list_label_paths = label_paths[:ln]
        else:
            self.list_camera_paths = camera_paths[ln:final_sample]
            self.list_label_paths = label_paths[ln:final_sample]

        return self._data_generator(self.base_path, 
                                    image_size=self.params['image_size'],
                                    out_size=self.params['out_size'], 
                                    list_camera_paths=self.list_camera_paths, 
                                    list_label_paths=self.list_label_paths)




    def _data_generator(self, base_path, image_size, out_size, list_camera_paths, list_label_paths):
            
        for camera_path, label_path in zip(list_camera_paths, list_label_paths):

                image = mpimg.imread(camera_path)
                w, h, _ = image.shape
                image = cv2.resize(image, (self.params['image_size'][1], self.params['image_size'][0]))

                with open(label_path) as json_file:
                    data = json.load(json_file)
                objs = data['objects']

                cars_mask = np.zeros((w, h))
                ground_mask = np.zeros((w, h))
                for i in range(len(objs)):
                    if objs[i]['label'] in ['car', 'truck']:
                        temp = np.array(objs[i]['polygon'])
                        rr, cc = polygon(temp[:, 0], temp[:, 1], (h, w))
                        cars_mask[cc, rr] = 1
                    elif objs[i]['label'] in ['road']:
                        temp = np.array(objs[i]['polygon'])
                        rr, cc = polygon(temp[:, 0], temp[:, 1], (h, w))
                        ground_mask[cc, rr] = 1
                    
                label_img_cars = cv2.resize(cars_mask.astype(np.float32), (out_size[1], out_size[0]))
                label_img_cars = label_img_cars > 0
                label_img_cars = label_img_cars.reshape(out_size).astype(np.float32)

                label_img_ground = cv2.resize(ground_mask.astype(np.float32), (out_size[1], out_size[0]))
                label_img_ground = label_img_ground > 0
                label_img_ground = label_img_ground.reshape(out_size).astype(np.float32)

                images = [image, image[:, ::-1, :]]
                labels_cars = [label_img_cars, label_img_cars[:, ::-1, :]]
                labels_ground = [label_img_ground, label_img_ground[:, ::-1, :]]

                
                temp_image = image.copy()
                temp_image[:, -100:, :] = 0
                temp_image[:, :100, :] = 0
                images.append(temp_image)

                temp_label = label_img_cars.copy()
                temp_label[:, -int((100/self.params['image_size'][0])*out_size[0]):, :] = 0
                temp_label[:, :int((100/self.params['image_size'][0])*out_size[0]), :] = 0
                labels_cars.append(temp_label)

                temp_label = label_img_ground.copy()
                temp_label[:, -int((100/self.params['image_size'][0])*out_size[0]):, :] = 0
                temp_label[:, :int((100/self.params['image_size'][0])*out_size[0]), :] = 0
                labels_ground.append(temp_label)

                temp_image = image.copy()
                temp_image[-50:, :, :] = 0
                temp_image[:50, :, :] = 0
                images.append(temp_image)

                temp_label = label_img_cars.copy()
                temp_label[-int((50/self.params['image_size'][1])*out_size[1]):, :, :] = 0
                temp_label[:int((50/self.params['image_size'][1])*out_size[1]), :, :] = 0
                labels_cars.append(temp_label)

                temp_label = label_img_ground.copy()
                temp_label[-int((50/self.params['image_size'][1])*out_size[1]):, :, :] = 0
                temp_label[:int((50/self.params['image_size'][1])*out_size[1]), :, :] = 0
                labels_ground.append(temp_label)

                yield (images, labels_cars, labels_ground)
