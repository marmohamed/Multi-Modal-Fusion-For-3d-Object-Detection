# coding: utf-8
# This script is modified from https://github.com/wizyoung/YOLOv3_TensorFlow

from __future__ import division, print_function

import numpy as np
import os
import glob
from utils.utils import *

def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    param:
        box: tuple or array, shifted to the origin (i. e. width, height and depth)
        clusters: numpy array of shape (k, 3) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    z = np.minimum(clusters[:, 2], box[2])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0 or np.count_nonzero(z == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y * z
    box_area = box[0] * box[1] * box[2]
    cluster_area = clusters[:, 0] * clusters[:, 1] * clusters[:, 2]

    iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    param:
        boxes: numpy array of shape (r, 3), where r is the number of rows
        clusters: numpy array of shape (k, 3) where k is the number of clusters
    return:
        average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    param:
        boxes: numpy array of shape (r, 3), where r is the number of rows
        k: number of clusters
        dist: distance function
    return:
        numpy array of shape (k, 3)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

def read_calib(calib_path):
    lines = []
    with open(calib_path) as calib_file:
        lines = calib_file.readlines()
    lines = list(filter(lambda x: len(x.split()) > 0, lines))
    calib_data = dict(list(map(lambda x: (x.split()[0][:-1], np.array(list(map(float, x.split()[1:])))), lines)))
    return calib_data


def parse_anno(annotation_path, calib_path):
    """
    the file format is as follows: 
    type, truncated, occluded, alpha, bbox_left, bbox_top, bbox_right, bbox_bottom,
    dimensions_height, dimensions_width, dimensions_length, location_x, location_y, location_z,
    rotation_y, score) 
    """
    lines = []
    with open(annotation_path) as label_file:
        lines = label_file.readlines()
    # filter car class
    lines = list(map(lambda x: x.split(), lines))
    lines = list(filter(lambda x: x[0] in ['Car', 'Van', 'Truck'] , lines))
    
    def get_parameter(index):
        return list(map(lambda x: x[index], lines))
    
    # TODO: crop the labels if the images are cropped
    classes = get_parameter(0)
    
    dimension_height = np.array(get_parameter(8)).astype(float)
    dimension_width = np.array(get_parameter(9)).astype(float)
    dimension_length = np.array(get_parameter(10)).astype(float)

    location_x = np.array(get_parameter(11)).astype(float)
    location_y = np.array(get_parameter(12)).astype(float)
    location_z = np.array(get_parameter(13)).astype(float)
    angles = np.array(get_parameter(14)).astype(float)

    calib_data = read_calib(calib_path)

    points = [project_point_from_camera_coor_to_velo_coor([location_x[i], location_y[i], location_z[i]], 
                                                        [dimension_height[i], dimension_width[i], dimension_length[i]],
                                                        angles[i],
                                                         calib_data)
                for i in range(len(classes))]

    x_range=(0, 70)
    y_range=(-40, 40)
    z_range=(-2.5, 1)
    size=(512, 448, 32)
    x_size = (x_range[1] - x_range[0])
    y_size = (y_range[1] - y_range[0])
    z_size = (z_range[1] - z_range[0])
            
    x_fac = (size[0]-1) / x_size
    y_fac = (size[1]-1) / y_size
    z_fac = (size[2]-1) / z_size
    import math
    for i in range(len(points)):
        b = points[i]
        x0 = b[0][0]
        y0 = b[0][1]
        x1 = b[1][0]
        y1 = b[1][1]
        x2 = b[2][0]
        y2 = b[2][1]
        u0 = -(x0) * x_fac + 512
        v0 = -(y0 + 40) * y_fac + 448
        u1 = -(x1) * x_fac + 512
        v1 = -(y1 + 40) * y_fac + 448
        u2 = -(x2) * x_fac + 512
        v2 = -(y2 + 40) * y_fac + 448
        # print(dimension_length[i])
        dimension_length[i] = math.sqrt((v1-v2)**2 + (u1-u2)**2)
        # print(dimension_length[i])
        dimension_width[i] = math.sqrt((v1-v0)**2 + (u1-u0)**2)
        # print(dimension_height[i])
        dimension_height[i] = math.sqrt((-(b[0][2]+2.5)*z_fac+(-b[4][2]-2.5)*z_fac)**2)
        # print(dimension_height[i])

    output = list(map(lambda i: [dimension_width[i], dimension_height[i], dimension_length[i]], range(len(dimension_height))))

    output = list(filter(lambda i: i[0] > 0 and i[1] > 0 and i[2] > 0, output))
    
    return output



def get_kmeans(anno, cluster_num=9):

    anchors = kmeans(anno, cluster_num)
    ave_iou = avg_iou(anno, anchors)

    # anchors = anchors.astype('int').tolist()
    anchors = anchors.tolist()

    anchors = sorted(anchors, key=lambda x: x[0] * x[1] * x[2])

    return anchors, ave_iou


if __name__ == '__main__':
    print('enter')
    annotation_paths = glob.glob('/Volumes/My Passport/Kitti_data/training/label_2/*.txt')
    calib_paths = glob.glob('/Volumes/My Passport/Kitti_data/data_object_calib/training/calib/*.txt')
    print('Reading files', len(annotation_paths), len(calib_paths))
    anno_result_ = np.array(list(map(lambda i: parse_anno(annotation_paths[i], calib_paths[i]), range(len(annotation_paths)))))
    anno_result = []
    for i in range(anno_result_.shape[0]):
        for j in range(len(anno_result_[i])):
            anno_result.append(anno_result_[i][j])
    anno_result = np.array(anno_result)
    print('Clustering', anno_result.shape)
    anchors, ave_iou = get_kmeans(anno_result, 2)

    anchor_string = ''
    for anchor in anchors:
        anchor_string += '{},{},{} , '.format(anchor[0], anchor[1], anchor[2])
    anchor_string = anchor_string[:-2]

    print('anchors are:')
    print(anchor_string)
    print('the average iou is:')
    print(ave_iou)

    # anchors are:
    # 9.330604225990367,30.05679515880936,26.936591286568465 , 11.687614364673774,43.868486378030084,37.42505613496754 
    # the average iou is:
    # 0.7313245856748651
    # the average anchor 10.5091093 , 36.96264077, 32.18082371

