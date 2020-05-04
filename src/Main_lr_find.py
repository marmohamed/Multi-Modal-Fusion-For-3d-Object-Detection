
from model import *
import os, random

from training.lr_finder import *
from training.detection_trainer import *
from data.detection_dataset_loader import *


if __name__ == '__main__':

    params = {
            'focal_loss': False,
            'fusion': True
        }
    model = Model(graph=None, **params)
    data_path = '../../Data/'
    dataset = DetectionDatasetLoader(data_path, None, 0.8, 0, True)
    bev_trainer = BEVDetectionTrainer(model, data_path, None)
    lr_finder = LRFinder(model, bev_trainer, data_path, dataset)

    list_files = list(map(lambda x: x.split('.')[0], os.listdir(data_path + 'data_object_image_3/training/image_3')))
    ln = int(len(list_files) * 0.8)
    loss_batch_size = 32
    lower_bound=1e-6
    upper_bound=1e-2

    epochs=2
    stepsPerEpoch = ln // loss_batch_size
    numBatchUpdates = epochs * stepsPerEpoch
    
    lr_mult = (upper_bound / lower_bound) ** (1.0 / numBatchUpdates)
    lr_finder.find(epochs=epochs, lower_bound=lower_bound, upper_bound=upper_bound, step_size=lr_mult, loss_batch_size=loss_batch_size)