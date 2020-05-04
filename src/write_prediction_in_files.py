
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf

import sys

from data.segmentation_dataset_loader import *
from data.detection_dataset_loader import *
from model import *
from Trainer import *
from evaluation.evaluate import *
from data.postprocessing.nms import *


def prepare_dataset_feed_dict(model, dataset, train_fusion_rgb, train_fusion_fv_lidar, use_nms):
        # camera_tensor, lidar_tensor, fv_velo_tensor, label_tensor, Tr_velo_to_cam, R0_rect, P3, shift_h, shift_w = sess.run(dataset)
        data = dataset.get_next(batch_size=1)

#         for i in range(len(data)):
#             data[i] = np.expand_dims(data[i], axis=0)
        camera_tensor, lidar_tensor, label_tensor, Tr_velo_to_cam, R0_rect, P3, shift_h, shift_w = data
#         print(np.max(camera_tensor))
        d = {model.train_inputs_rgb: camera_tensor,
                model.train_inputs_lidar: lidar_tensor,
                model.Tr_velo_to_cam: Tr_velo_to_cam,
                model.R0_rect: R0_rect,
                model.P3: P3,
                model.shift_h: shift_h,
                model.shift_w: shift_w,
                model.use_nms: use_nms,
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
            anchor = np.array([x+0.5, y+0.5, 0.5, anchors[0], anchors[1], anchors[2]])
#             if not is_label:
#               out[:3] = sigmoid(out[:3])
            out[:3] = np.tanh(out[:3])*0.5 * anchor[3:6] + anchor[:3]
            
            out[:2] = out[:2] * ratio
            out[2] = out[2] * 32
            
             out[3:6] = np.square(np.maximum(0, out[3:6])) * anchors
            
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
                size=(512, 448, 32), th=0.5):
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



def read_label2(label_path, calib_path, shift_h, shift_w, x_range=(0, 71), y_range=(-40, 40), z_range=(-3.0, 1), 
                    size=(512, 448, 32), get_actual_dims=False, from_file=True, translate_x=0, translate_y=0, translate_z=0, ang=0, get_neg=False):

    """
    the file format is as follows: 
    type, truncated, occluded, alpha, bbox_left, bbox_top, bbox_right, bbox_bottom,
    dimensions_height, dimensions_width, dimensions_length, location_x, location_y, location_z,
    rotation_y, score) 
    """
    if from_file:
        lines = []
        with open(label_path) as label_file:
            lines = label_file.readlines()
    else:
        lines = label_path.split('\n')
    # filter car class
    lines = list(map(lambda x: x.split(), lines))
    if len(lines) > 0:
        if get_neg:
            lines = list(filter(lambda x: len(x) > 0 and ( x[0] not in ['Car', 'Van', 'Truck', 'Tram', 'DontCare']), lines))
            if len(lines) > 0:
                lines = lines[:1]
        else:
            lines = list(filter(lambda x: len(x) > 0 and ( x[0] in ['Car', 'Van', 'Truck', 'Tram']), lines))
    
    def get_parameter(index):
        return list(map(lambda x: x[index], lines))
    
    classes = np.array(get_parameter(0))
    dimension_height = np.array(get_parameter(8)).astype(float)
    dimension_width = np.array(get_parameter(9)).astype(float)
    dimension_length = np.array(get_parameter(10)).astype(float)
    # TODO: take shift into consideration - URGENT
    location_x = np.array(get_parameter(11)).astype(float)
    location_y = np.array(get_parameter(12)).astype(float)
    location_z = np.array(get_parameter(13)).astype(float)
    angles = np.array(get_parameter(14)).astype(float)
    directions = np.array(angles>= 0).astype(float)
    
    # print(len(classes))
    calib_data = read_calib(calib_path)

    locations = np.array([[location_x[i], location_y[i], location_z[i]] for i in range(len(classes))])
    # print(locations)
    if len(locations) > 0 and len(locations[0]) > 0:
        locations = project_rect_to_velo(locations, calib_data['R0_rect'].reshape((3, 3)), calib_data['Tr_velo_to_cam'].reshape((3, 4)))
    # print(locations)
    # print(z_range)

    indx = []
    i = 0
    for point in locations:
        if (point[0] >= x_range[0]  and point[0] <= x_range[1])\
            and (point[1] >= y_range[0] and point[1] <= y_range[1])\
            and (point[2] >= z_range[0] and point[2] <= z_range[1]):
            indx.append(i)
        i += 1

    indxes = np.array(list(map(lambda point: (point[0] >= x_range[0]  and point[0] <= x_range[1])
                                    and (point[1] >= y_range[0] and point[1] <= y_range[1])
                                    and (point[2] >= z_range[0] and point[2] <= z_range[1]) , locations)))
    locations = np.array(list(filter(lambda point: (point[0] >= x_range[0]  and point[0] <= x_range[1])
                                    and (point[1] >= y_range[0] and point[1] <= y_range[1])
                                    and (point[2] >= z_range[0] and point[2] <= z_range[1]) , locations)))

    if len(indx) > 0:
        dimension_height = dimension_height[indx]
        dimension_width = dimension_width[indx]
        dimension_length = dimension_length[indx]
        location_x = location_x[indx]
        location_y = location_y[indx]
        location_z = location_z[indx]
        angles = angles[indx]
        classes = classes[indx]
        directions = directions[indx]

    if len(locations) > 0:
        locations[:, :3] = locations[:, :3] - np.array([translate_x, translate_y, -translate_z])

    # print('.......')
    # print(len(locations))

    points = [project_point_from_camera_coor_to_velo_coor([location_x[i], location_y[i], location_z[i]], 
                                                        [dimension_height[i], dimension_width[i], dimension_length[i]],
                                                        angles[i],
                                                         calib_data)
                for i in range(len(locations))]
    
    x_size = (x_range[1] - x_range[0])
    y_size = (y_range[1] - y_range[0])
    z_size = (z_range[1] - z_range[0])
            
    x_fac = (size[0]-1) / x_size
    y_fac = (size[1]-1) / y_size
    z_fac = (size[2]-1) / z_size
    if get_actual_dims:
        import math
        for i in range(len(points)):
            b = points[i]
            x0 = b[0][0]
            y0 = b[0][1]
            x1 = b[1][0]
            y1 = b[1][1]
            x2 = b[2][0]
            y2 = b[2][1]
            u0 = -(x0) * x_fac + size[0]
            v0 = -(y0 + 40) * y_fac + size[1]
            u1 = -(x1) * x_fac + size[0]
            v1 = -(y1 + 40) * y_fac + size[1]
            u2 = -(x2) * x_fac + size[0]
            v2 = -(y2 + 40) * y_fac + size[1]
            dimension_length[i] = math.sqrt((v1-v2)**2 + (u1-u2)**2)
            dimension_width[i] = math.sqrt((v1-v0)**2 + (u1-u0)**2)
            dimension_height[i] = math.sqrt((-(b[0][2]+(-1*z_range[1]))*z_fac+(-b[4][2]+z_range[1])*z_fac)**2)

      
    for i in range(len(locations)):
        if angles[i] < 0:
            angles[i] += 3.14

    x_range = (x_range[0] + translate_x, x_range[1] + translate_x)
    y_range = (y_range[0] + translate_y, y_range[1] + translate_y)
    z_range = (z_range[0] + translate_z, z_range[1] + translate_z)
    output = [[-(locations[i][0] + -1*x_range[0]) * x_fac + size[0], -(locations[i][1] + -1*y_range[0]) * y_fac + size[1], -(locations[i][2] + -1*z_range[0]) * z_fac + size[2], 
                dimension_length[i], dimension_width[i], dimension_height[i], angles[i]] 
                for i in range(len(locations))]
    # import math
    if ang != 0:
        for i in range(len(locations)):
            w = size[0]
            h = size[1]
            output[i][0], output[i][1] = rotate2((w//2, h//2), (output[i][0], output[i][1]), ang / 57.2958)
            output[i][6] = output[i][6] - ang / 57.2958

    output = list(filter(lambda point: 0 <= point[0] < size[0] and 0 <= point[1] < size[1] and 0 <= point[2] < size[2] , output))
    output = np.array(output)

    if from_file:
        return points, output, calib_data['Tr_velo_to_cam'], calib_data['R0_rect'], calib_data['P2'], directions
    else:
        return output, indxes


def write_predictions(final_output, th, new_file_path, current_file, apply_nms=False, sess=None, base_path = '../../Data'):
    converted_points = convert_prediction_into_real_values(final_output[0, :, :, :, :], th=th)
    points = get_points(converted_points, base_path + '/data_object_calib/training/calib/'+ current_file + '.txt', th=th)
    res = '\n'.join([' '.join([str(l) for l in points[i]]) for i in range(len(points))])
    if apply_nms:
        labels, indxes = read_label2(res, base_path + '/data_object_calib/training/calib/'+ current_file + '.txt', 0, 0, get_actual_dims=True, from_file=False)
        if len(labels) != len(points):
            print('not the same', new_file_path)
#             return
        if len(labels) > 0:
            points = np.array(points)
            points = points[indxes]
            selected_idx = nms(labels, np.array([points[i][-1] for i in range(len(points))]), max_output_size=100, iou_threshold=0.3, sess=sess)
        else:
            selected_idx = []
        points = np.array(points)
        if len(selected_idx) > 0:
            points = points[selected_idx]
            res = '\n'.join([' '.join([str(l) for l in points[i]]) for i in range(len(points))])
        else:
            res=""
    text_file = open(new_file_path, "wb+")
    text_file.write(res.encode())
    text_file.close()



def eval(dir_name, model, list_files, best=False, apply_nms=False):
    with model.graph.as_default():
            
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            if best:
                model.saver.restore(sess, tf.train.latest_checkpoint('./training_files/tmp_best2/'))
            else:
                model.saver.restore(sess, tf.train.latest_checkpoint('./training_files/tmp/'))
            dataset = DetectionDatasetLoader(base_path='../../Data', training_per=0.5, batch_size=1, random_seed=0, training=False)
        
            cls_losses = []
            reg_losses = []
            total_losses = []
            i = 0
                        
            try:    
                while True:
                    feed_dict = prepare_dataset_feed_dict(model, dataset, False, False, False)
                    final_output= sess.run(model.final_output, feed_dict=feed_dict)
                    if i < len(list_files):
                        current_file = list_files[i]
                        th = 0.10
                        new_file_path = './prediction_files/' + dir_name + '/bev/th10_2/data/' + current_file + '.txt'
                        write_predictions(final_output, th, new_file_path, current_file, apply_nms=apply_nms, sess=sess)
                        th = 0.20
                        new_file_path = './prediction_files/' + dir_name + '/bev/th20_2/data/' + current_file + '.txt'
                        write_predictions(final_output, th, new_file_path, current_file, apply_nms=apply_nms)
                        th = 0.30
                        new_file_path = './prediction_files/' + dir_name + '/bev/th30_2/data/' + current_file + '.txt'
                        write_predictions(final_output, th, new_file_path, current_file, apply_nms=apply_nms)
                        th = 0.40
                        new_file_path = './prediction_files/' + dir_name + '/bev/th40_2/data/' + current_file + '.txt'
                        write_predictions(final_output, th, new_file_path, current_file, apply_nms=apply_nms)
                        th = 0.50
                        new_file_path = './prediction_files/' + dir_name + '/bev/th50_2/data/' + current_file + '.txt'
                        write_predictions(final_output, th, new_file_path, current_file, apply_nms=apply_nms)

                    else:
                        break
                    i += 1
                    if i % 100 == 0:
                        print('i = ', i)
                               
            except tf.errors.OutOfRangeError:
                pass
            except StopIteration:
                pass


if __name__ == '__main__':

    base_path = '../../Data'
    list_files = list(map(lambda x: x.split('.')[0], os.listdir(base_path+'/data_object_image_3/training/image_3')))
    random.seed(0)
    random.shuffle(list_files)
    ln = int(len(list_files) * 0.5)
    list_files= list_files[ln:]

    params = {
        'fusion': False
    }
    model = Model(graph=None, **params)
        
    try:
        print('Start best')
        dir_name = 'predictions_focal_model_new_11_best'
        eval(dir_name, model, list_files, best=True, apply_nms=False)
    except:
        print('error')

    try:
        print('Start')
        dir_name = 'predictions_focal_model_new_11'
        eval(dir_name, model, list_files, best=False, apply_nms=False)
    except:
        print('error')

    try:
        print('Start best nms')
        dir_name = 'predictions_focal_model_new_11_best_nms'
        eval(dir_name, model, list_files, best=True, apply_nms=True)
    except:
        print('error')

    try:
        print('Start nms')
        dir_name = 'predictions_focal_model_new_11_nms'
        eval(dir_name, model, list_files, best=False, apply_nms=True)
    except:
        print('error')



