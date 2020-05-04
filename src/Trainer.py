import tensorflow as tf
import argparse
import numpy as np
from tensorflow.python import debug as tf_debug

from utils.constants import *
from utils.utils import *
from utils.anchors import *
from utils.nms import *
from loss.losses import *
from models.ResNetBuilder import *
from models.ResnetImage import *
from models.ResnetLidarBEV import *
from models.ResnetLidarFV import *
from FPN.FPN import *
from Fusion.FusionLayer import *
from training.Trainer import *
from training.segmentation_trainer import *
from training.detection_trainer import *




class ModelTrainer(object):

    def __init__(self, model, data_base_path):
        self.model = model
        self.data_base_path = data_base_path

        self.segmentation_trainer = SegmentationTrainer(self.model, self.data_base_path, None)
        self.fusion_trainer = FusionDetectionTrainer(self.model, self.data_base_path, None)
        self.bev_trainer = BEVDetectionTrainer(self.model, self.data_base_path, None)

    def train_bev(self, **kwargs):
        d = {'num_summary_images': kwargs['num_summary_images']}
        self.bev_trainer.train(restore=kwargs['restore'], 
                    epochs=kwargs['epochs'], 
                    num_samples=kwargs['num_samples'], 
                    training_per=kwargs['training_per'], 
                    random_seed=kwargs['random_seed'], 
                    training=kwargs['training'], 
                    batch_size=kwargs['batch_size'], 
                    save_steps=kwargs['save_steps'],
                    start_epoch=kwargs['start_epoch'],
                    **d)

    def train_fusion(self, **kwargs):
        d = {'num_summary_images': kwargs['num_summary_images']}
        self.fusion_trainer.train(restore=kwargs['restore'], 
                    epochs=kwargs['epochs'], 
                    num_samples=kwargs['num_samples'], 
                    training_per=kwargs['training_per'], 
                    random_seed=kwargs['random_seed'], 
                    training=kwargs['training'], 
                    batch_size=kwargs['batch_size'], 
                    save_steps=kwargs['save_steps'],
                    start_epoch=kwargs['start_epoch'],
                    **d)

    def train_images_seg(self, **kwargs):
        d = {
            'epochs_img_head': kwargs['epochs_img_head'],
            'epochs_img_all': kwargs['epochs_img_all'],
            'segmentation_kitti': kwargs['segmentation_kitti'],
            'segmentation_cityscapes': kwargs['segmentation_cityscapes'],
            'num_summary_images': kwargs['num_summary_images']
        }
        self.segmentation_trainer.train(restore=kwargs['restore'], 
                    epochs=kwargs['epochs'], 
                    num_samples=kwargs['num_samples'], 
                    training_per=kwargs['training_per'], 
                    random_seed=kwargs['random_seed'], 
                    training=kwargs['training'], 
                    batch_size=kwargs['batch_size'], 
                    save_steps=kwargs['save_steps'],
                    start_epoch=kwargs['start_epoch'],
                    **d)
 
    
