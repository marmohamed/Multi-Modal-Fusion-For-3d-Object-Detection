

import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf
from data.segmentation_dataset_loader import *
from data.detection_dataset_loader import *
from model import *
from Trainer import *
from evaluation.evaluate import *
from data.postprocessing.nms import *



def prepare_dataset_feed_dict(model, dataset, train_fusion_rgb, train_fusion_fv_lidar, anchor_values, use_nms):
        data = dataset.get_next(batch_size=1)

        camera_tensor, lidar_tensor, label_tensor = data
        d = {model.train_inputs_rgb: camera_tensor,
                model.train_inputs_lidar: lidar_tensor,
                model.y_true: label_tensor,
                model.train_fusion_rgb: train_fusion_rgb,
                model.is_training: False}
        return d


def sigmoid(x):
  return (1 / (1 + np.exp(-x.astype(np.float128)))).astype(np.float32)

def convert_prediction_into_real_values(label_tensor, 
            anchors=np.array([3.9, 1.6, 1.5]), 
            input_size=(512, 448), output_size=(128, 112), is_label=False, th=0.5):

    ratio = input_size[0] // output_size[0]
    result = []
    ones_index = np.where(sigmoid(label_tensor[:, :, :, -1])>=th)
    if len(ones_index) > 0 and len(ones_index[0]) > 0:
        for i in range(0, len(ones_index[0]), 1):
            x = ones_index[0][i]
            y = ones_index[1][i]
            
            out = np.copy(label_tensor[ones_index[0][i], ones_index[1][i], ones_index[2][i], :])
            anchor = np.array([x+0.5, y+0.5, 1., anchors[0], anchors[1], anchors[2]])

            out[:3] = out[:3] * anchor[3:6] + anchor[:3]
            
            out[:2] = out[:2] * ratio
            out[2] = out[2] * 40
            
            out[3:6] = np.exp(out[3:6]) * anchors
            
            k = ones_index[2][i]
            if not is_label:
              out[6] = sigmoid(out[6]) * np.pi/2 - np.pi/4
            if k == 0 and out[6] < 0:
                out[6] = out[6] + np.pi
                
            out[6] = out[6] + k * (np.pi/2)
                        
            result.append(out)
            
    return np.array(result)


def get_points(converted_points, calib_path, 
                x_range=(0, 71), y_range=(-40, 40), z_range=(-3.0, 1), 
                size=(512, 448, 40), th=0.5):
    all_result = []
    for converted_points_ in converted_points:
        if sigmoid(converted_points_[-1]) >= th:
#              and sigmoid(converted_points_[8]) < th2
            result = [0] * 16
            result[0] = 'Car'
            result[1] = -1
            result[2] = -1
            result[3] = -10
            result[8] = converted_points_[5]
            result[9] = converted_points_[4]
            result[10] = converted_points_[3]
            result[14] = converted_points_[6]
            result[15] = sigmoid(converted_points_[-1])

            calib_data = read_calib(calib_path)

            # x_range=(0, 70)
            # y_range=(-40, 40)
            # z_range=(-2.5, 1)

            x_size = (x_range[1] - x_range[0])
            y_size = (y_range[1] - y_range[0])
            z_size = (z_range[1] - z_range[0])

            x_fac = (size[0]-1) / x_size
            y_fac = (size[1]-1) / y_size
            z_fac = (size[2]-1) / z_size

            x, y, z = -((converted_points_[:3] - size) / np.array([x_fac, y_fac, z_fac])) - np.array([0, -1*y_range[0], -1*z_range[0]]) 
            point = np.array([[x, y, z]])
            box3d_pts_3d = point

            pts_3d_ref = project_velo_to_ref(box3d_pts_3d, calib_data['Tr_velo_to_cam'].reshape((3, 4)))
            pts_3d_ref = project_ref_to_rect(pts_3d_ref, calib_data['R0_rect'].reshape((3, 3)))[0]
            for k in range(3):
                result[11 + k] = pts_3d_ref[k]

            imgbbox = ProjectTo2Dbbox(pts_3d_ref, converted_points_[5], converted_points_[4],
                         converted_points_[3], converted_points_[6], calib_data['P2'].reshape((3, 4)))

            result[4:8] = imgbbox
            all_result.append(result)
    return all_result


def nms2(label, scores, max_output_size=100, iou_threshold=0.1, sess=None):
    boxes = []

    for j in range(0, len(label)):

        w = label[j][3]
        h = label[j][4] 
        x = label[j][0]
        y = label[j][1]
        a = label[j][6]
        
        polygon = convert5Pointto8Point(y, x, w, h, -a*57.2958)
        xs = polygon[0::2]
        ys = polygon[1::2]
            
        boxes.append([xs[0], ys[0], xs[2], ys[2]])

    boxes = np.array(boxes)
    with tf.Graph().as_default():
        selected_indices = tf.image.non_max_suppression(
                    boxes, scores, max_output_size=max_output_size, iou_threshold=iou_threshold)
        if sess is not None:
            selected_indices = sess.run(selected_indices)
        else:
            with tf.Session() as sess:
                selected_indices = sess.run(selected_indices)

    return selected_indices


def write_predictions(final_output, th, new_file_path, apply_nms=False, sess=None):
    converted_points = convert_prediction_into_real_values(final_output[0, :, :, :, :], th=th)
    points = get_points(converted_points, base_path + '/data_object_calib/training/calib/'+ current_file + '.txt', th=th)
    res = '\n'.join([' '.join([str(l) for l in points[i]]) for i in range(len(points))])

    if apply_nms:
            labels, indxes = read_label2(res, base_path + '/data_object_calib/training/calib/'+ current_file + '.txt', 0, 0, get_actual_dims=True, from_file=False)
            points = np.array(points)
            
            if len(labels) > 0:
                points = points[indxes]
                selected_idx = nms(labels, np.array([points[i][-1] for i in range(len(points))]), max_output_size=100, iou_threshold=0.3, sess=sess)
            else:
                selected_idx = []

            if len(selected_idx) > 0:
                points = points[selected_idx]
                res = '\n'.join([' '.join([str(l) for l in points[i]]) for i in range(len(points))])
            else:
                res=""

    text_file = open(new_file_path, "wb+")
    text_file.write(res.encode())
    text_file.close()



