# many methods are imported from https://github.com/kuixu/kitti_object_vis

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import numpy as np
import random
from scipy import misc
import math

def rotate2(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def project_to_image(pts_3d, P):
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n,1))))
    #print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P)) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    return pts_2d[:,0:2]

def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

def cart2hom(pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom
def project_ref_to_velo(pts_3d_ref, Tr_velo_to_cam):
        pts_3d_ref = cart2hom(pts_3d_ref) # nx4
        C2V = inverse_rigid_trans(Tr_velo_to_cam.reshape((3, 4)))
        return np.dot(pts_3d_ref, np.transpose(C2V))

def project_rect_to_ref(pts_3d_rect, R0_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(R0_rect.reshape((3, 3))), np.transpose(pts_3d_rect)))


def project_point_from_camera_coor_to_velo_coor(location, dimemsion, agnle, calib_data):
    R = roty(agnle)
    h, w, l = dimemsion
    x, y, z = location
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [0,0,0,0,-h,-h,-h,-h];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    #print corners_3d.shape
    corners_3d[0,:] = corners_3d[0,:] + x;
    corners_3d[1,:] = corners_3d[1,:] + y;
    corners_3d[2,:] = corners_3d[2,:] + z;
    #print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
#     if np.any(corners_3d[2,:]<0.1):
#         corners_2d = None
#         return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
#     corners_2d = project_to_image(np.transpose(corners_3d), calib_data['P3'].reshape((3, 4)));
    box3d_pts_3d = np.transpose(corners_3d)

    pts_3d_ref = project_rect_to_ref(box3d_pts_3d, calib_data['R0_rect'])
    result = project_ref_to_velo(pts_3d_ref, calib_data['Tr_velo_to_cam'])
#     print(result)
    return result

def project_point_from_lidar_to_image(point, Tr_velo_to_cam, R0_rect, P3, shift_h, shift_w):
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

    point = [-1, -1, -1, 1] * point / [x_fac, y_fac, z_fac, 1] + [0, -40, -2.5, 0] - [512, 448, 32, 0]

    point = tf.reshape(point, (-1, 4, 1))
    x = tf.matmul(Tr_velo_to_cam, tf.cast(point, tf.float32))
    rgb_point = tf.matmul(P3, tf.matmul(R0_rect, x))
    rgb_point = rgb_point[:, :2, 0] / rgb_point[:, 2, 0]
    temp = tf.concat([shift_h, shift_w], 1)
    rgb_point = rgb_point + temp
    return rgb_point

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)