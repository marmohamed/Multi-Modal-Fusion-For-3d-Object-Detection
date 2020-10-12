import numpy as np
import math
from scipy.ndimage import rotate
import cv2
from data.data_utils.fv_utils import *

def in_range_points(points, x, y, z, x_range, y_range, z_range):
    """ 
    This function is imported from https://github.com/windowsub0406/KITTI_Tutorial/blob/master/Convert_Velo_2_Topview_detail.ipynb
    extract in-range points 
    """
    return points[np.logical_and.reduce((x >= x_range[0], x <= x_range[1], y >= y_range[0], \
                                         y <= y_range[1], z >= z_range[0], z <= z_range[1]))]

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def velo_points_bev(rot, tr, sc, lidar_path, calib_path, x_range=(0, 71), y_range=(-40, 40), z_range=(-3.0, 1), 
                    size=(512, 448, 40), img_width=1224, img_height=370, translate_x=0, translate_y=0, translate_z = 0, ang=0, scale=0, fliplr=False):
    """
    # TODO :  8-neighbor interpolation
    """
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    calib = Calibration(calib_path)
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(points[:, :3],
        calib, 0, 0, img_width, img_height, True)
    # points = imgfov_pc_velo
    points = points[fov_inds, :]
    points = (points.transpose() + tr).transpose()
    points = np.matmul(rot, points.transpose()).transpose()
    points = np.matmul(sc, points.transpose()).transpose()
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]

    x_lim = in_range_points(x, x, y, z, x_range, y_range, z_range)
    y_lim = in_range_points(y, x, y, z, x_range, y_range, z_range)
    z_lim = in_range_points(z, x, y, z, x_range, y_range, z_range)
    i_lim = in_range_points(intensity, x, y, z, x_range, y_range, z_range)

#     x_lim -= translate_x
#     y_lim -= translate_y
#     z_lim -= translate_z
    x_range = (x_range[0] + translate_x, x_range[1] + translate_x)
    y_range = (y_range[0] + translate_y, y_range[1] + translate_y)
    z_range = (z_range[0] + translate_z, z_range[1] + translate_z)

    x_size = (x_range[1] - x_range[0])
    y_size = (y_range[1] - y_range[0])
    z_size = (z_range[1] - z_range[0])
        
    x_fac = (size[0]-1) / x_size
    y_fac = (size[1]-1) / y_size
    z_fac = (size[2]-1) / z_size

    # if x_range[0] < 0:
    x_lim = x_lim + -1*x_range[0]
    # if y_range[0] < 0:
    y_lim = y_lim + -1*y_range[0]
    # if z_range[0] < 0:
    z_lim = z_lim + -1*z_range[0]
        
    x_lim = -1 * (x_lim * x_fac).astype(np.int32)
    y_lim = -1 * (y_lim * y_fac).astype(np.int32)
    z_lim = 1 * (z_lim * z_fac).astype(np.int32)

    x_lim2 = x_lim[:]
    y_lim2 = y_lim[:]
    z_lim2 = z_lim[:]
    i_lim2 = i_lim[:]

    x_lim = x_lim[(x_lim2>-size[0]) & (x_lim2<= 0) & (y_lim2>-size[1]) & (y_lim2 <= 0) & (z_lim2<size[2]) & (z_lim2 >= 0)]
    y_lim = y_lim[(x_lim2>-size[0]) & (x_lim2<= 0) & (y_lim2>-size[1]) & (y_lim2 <= 0) & (z_lim2<size[2]) & (z_lim2 >= 0)]
    z_lim = z_lim[(x_lim2>-size[0]) & (x_lim2<= 0) & (y_lim2>-size[1]) & (y_lim2 <= 0) & (z_lim2<size[2]) & (z_lim2 >= 0)]
    i_lim = i_lim[(x_lim2>-size[0]) & (x_lim2<= 0) & (y_lim2>-size[1]) & (y_lim2 <= 0) & (z_lim2<size[2]) & (z_lim2 >= 0)]
    
    
    img = np.zeros([size[0], size[1], size[2]+1], dtype=np.float32)
    # occupancy grid
    img[x_lim, y_lim, z_lim] = 255.
    img[x_lim, y_lim, -1] = i_lim * 255.
    if fliplr:
        img = np.fliplr(img)
    return img
