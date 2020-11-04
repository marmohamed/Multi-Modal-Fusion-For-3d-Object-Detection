import numpy as np
import math
from data.data_utils.reader_utils import read_calib
import tensorflow as tf
from data.detection_dataset_loader import *


def cart2hom(pts_3d):
    ''' Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    '''
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
    return pts_3d_hom

def project_velo_to_ref(pts_3d_velo, Tr_velo_to_cam):
    pts_3d_velo = cart2hom(pts_3d_velo) # nx4
    return np.dot(pts_3d_velo, np.transpose(Tr_velo_to_cam))

    
def project_ref_to_rect(pts_3d_ref, R0_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(R0_rect, np.transpose(pts_3d_ref)))

def ProjectTo2Dbbox(center, h, w, l, r_y, P2):
    # input: 3Dbbox in (rectified) camera coords

    Rmat = np.asarray([[math.cos(r_y), 0, math.sin(r_y)],
                       [0, 1, 0],
                       [-math.sin(r_y), 0, math.cos(r_y)]],
                       dtype='float32')

    p0 = center + np.dot(Rmat, np.asarray([l/2.0, 0, w/2.0], dtype='float32').flatten())
    p1 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, w/2.0], dtype='float32').flatten())
    p2 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, -w/2.0], dtype='float32').flatten())
    p3 = center + np.dot(Rmat, np.asarray([l/2.0, 0, -w/2.0], dtype='float32').flatten())
    p4 = center + np.dot(Rmat, np.asarray([l/2.0, -h, w/2.0], dtype='float32').flatten())
    p5 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, w/2.0], dtype='float32').flatten())
    p6 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, -w/2.0], dtype='float32').flatten())
    p7 = center + np.dot(Rmat, np.asarray([l/2.0, -h, -w/2.0], dtype='float32').flatten())

    points = np.array([p0, p1, p2, p3, p4, p5, p6, p7])

    points_hom = np.ones((points.shape[0], 4)) # (shape: (8, 4))
    points_hom[:, 0:3] = points

    # project the points onto the image plane (homogeneous coords):
    img_points_hom = np.dot(P2, points_hom.T).T # (shape: (8, 3)) (points_hom.T has shape (4, 8))
    # normalize:
    img_points = np.zeros((img_points_hom.shape[0], 2)) # (shape: (8, 2))
    img_points[:, 0] = img_points_hom[:, 0]/img_points_hom[:, 2]
    img_points[:, 1] = img_points_hom[:, 1]/img_points_hom[:, 2]

    u_min = np.min(img_points[:, 0])
    v_min = np.min(img_points[:, 1])
    u_max = np.max(img_points[:, 0])
    v_max = np.max(img_points[:, 1])

    left = int(u_min)
    top = int(v_min)
    right = int(u_max)
    bottom = int(v_max)

    projected_2Dbbox = [left, top, right, bottom]

    return projected_2Dbbox


def sigmoid(x):
    x = x.astype(np.float128)
    x = 1 / (1 + np.exp(-x))
    x = x.astype(np.float32)
    return x

def convert_prediction_into_real_values(label_tensor, truth_value=None,
            anchors=np.array([3.9, 1.6, 1.5]), 
            input_size=(512, 448), output_size=(128, 112), is_label=False, th=0.5):

    ratio = input_size[0] // output_size[0]
    result = []
    if not is_label:
        ones_index = np.where(sigmoid(label_tensor[:, :, :, -1])>=th)
    else:
        ones_index = np.where(label_tensor[:, :, :, -1]>=th)
    if truth_value is not None:
        ones_index = np.where(truth_value[:, :, :, -1]>=th)
#     print(ones_index)
    if len(ones_index) > 0 and len(ones_index[0]) > 0:
        for i in range(0, len(ones_index[0]), 1):
            x = ones_index[0][i]
            y = ones_index[1][i]
            
            out = np.copy(label_tensor[ones_index[0][i], ones_index[1][i], ones_index[2][i], :])
            anchor = np.array([x+0.5, y+0.5, 1., anchors[0], anchors[1], anchors[2]])

            out[:3] = sigmoid(out[:3])
            out[:2] = out[:2] - 0.5 + anchor[:2]
            
            out[:2] = out[:2] * ratio
            out[2] = out[2] * 40
            
            out[3:6] = sigmoid(out[3:6]) * 3 * anchors
            
            k = ones_index[2][i]
            if not is_label:
              out[6] = sigmoid(out[6]) * np.pi/2 - np.pi/4
            else:
                out[6] = out[6]
            
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

            
            calib_reader = CalibReader(calib_path)
            calib_data = calib_reader.read_calib()

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


def prepare_dataset_feed_dict(model, dataset, train_fusion_rgb):
        data = dataset.get_next(batch_size=1)
        camera_tensor, lidar_tensor, label_tensor= data
        d = {model.train_inputs_rgb: camera_tensor,
                model.train_inputs_lidar: lidar_tensor,
                model.y_true: label_tensor,
                model.train_fusion_rgb: train_fusion_rgb,
                model.is_training: False,
                model.weight_cls: 1,
                model.weight_dim: 1,
                model.weight_loc: 1,
                model.weight_theta: 1}
        return d


def write_predictions(labels_output, calib_path, new_file_path, th=0.5, truth_value=None, is_label=False):
    converted_points = convert_prediction_into_real_values(labels_output, truth_value=truth_value, th=th, is_label=is_label)
    points = get_points(converted_points, calib_path, th=th)
    res = '\n'.join([' '.join([str(l) for l in points[i]]) for i in range(len(points))])
    text_file = open(new_file_path, "wb+")
    text_file.write(res.encode())
    text_file.close()

def write_all_predictions(model, dir_name, training, augment=False, get_best=False, fusion=False):
    with model.graph.as_default():
            
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            if get_best:
                model.saver.restore(sess, tf.train.latest_checkpoint('../training_files/tmp_best_2/'))
            else:
                model.saver.restore(sess, tf.train.latest_checkpoint('../training_files/tmp/'))

            dataset = DetectionDatasetLoader(base_path='../../../Data', training_per=0.5, batch_size=1, random_seed=0, training=training, augment=augment)
        
            cls_losses = []
            reg_losses = []
            total_losses = []
            i = 0
            
            apply_nms=False

            if training:
                file_name = '/trainsplit.txt'
            else:
                            file_name = '/valsplit.txt'
            base_path = '../../../Data'
            with open(base_path + file_name, 'r') as f:
                            list_file_nums = f.readlines()
            list_files = ['0'*(6-len(l.strip())) + l.strip() for l in list_file_nums]
            list_calib_paths = list(map(lambda x: base_path + '/data_object_calib/training/calib/' + x + '.txt', list_files))
            
            try:    
                while True:
                    feed_dict = prepare_dataset_feed_dict(model, dataset, fusion)
                    final_output= sess.run(model.final_output, feed_dict=feed_dict)

                    if i < len(list_files):
                        current_file = list_files[i]

                        for th, th_str in zip([0.05, 0.1, 0.2, 0.3, 0.4, 0.5], ['05', '10', '20', '30', '40', '50']):
                            new_file_path = '../prediction_files/' + dir_name + '/bev/th' + th_str + '_2/data/' + current_file + '.txt'
                            write_predictions(final_output, list_calib_paths[i], new_file_path, th=th)
                            
                    else:
                        break

                    i += 1
                    if i % 100 == 0:
                        print('i = ', i)
            except tf.errors.OutOfRangeError:
                pass
            except StopIteration:
                pass
            finally:
                print('Done')