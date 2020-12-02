import numpy as np
import math
from scipy.ndimage import rotate
import cv2
from data.data_utils.fv_utils import *

class LidarReader:


    def __init__(self, lidar_path, calib_path, image_path, rot, tr, sc, fliplr=False, interpolate=True,
                        x_range=(0, 70), 
                        y_range=(-40, 40), 
                        z_range=(-2.5, 1), 
                        size=(448, 512, 40)):
        self.lidar_path = lidar_path
        self.calib_path = calib_path
        self.image_path = image_path
        self.rot = rot
        self.sc = sc
        self.tr = tr
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.size = size
        self.interpolate = interpolate
        self.fliplr=fliplr



    def read_lidar(self):

        points = np.fromfile(self.lidar_path, dtype=np.float32).reshape(-1, 4)
        calib = Calibration(self.calib_path)
        img_height, img_width, _ = cv2.imread(self.image_path).shape
        imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(points[:, :3],
            calib, 0, 0, img_width, img_height, True)

        points = points[fov_inds, :]
        points = (points.transpose() + self.tr).transpose()
        points = np.matmul(self.rot, points.transpose()).transpose()
        points = np.matmul(self.sc, points.transpose()).transpose()
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        intensity = points[:, 3]

        x_lim = self.in_range_points(x, x, y, z, self.x_range, self.y_range, self.z_range)
        y_lim = self.in_range_points(y, x, y, z, self.x_range, self.y_range, self.z_range)
        z_lim = self.in_range_points(z, x, y, z, self.x_range, self.y_range, self.z_range)
        i_lim = self.in_range_points(intensity, x, y, z, self.x_range, self.y_range, self.z_range)

        x_size = (self.x_range[1] - self.x_range[0])
        y_size = (self.y_range[1] - self.y_range[0])
        z_size = (self.z_range[1] - self.z_range[0])
            
        x_fac = (self.size[0]-1) / x_size
        y_fac = (self.size[1]-1) / y_size
        z_fac = (self.size[2]-1) / z_size

        # if x_range[0] < 0:
        x_lim = x_lim + -1*self.x_range[0]
        # if y_range[0] < 0:
        y_lim = y_lim + -1*self.y_range[0]
        # if z_range[0] < 0:
        z_lim = z_lim + -1*self.z_range[0]
            
        x_lim = -1 * (x_lim * x_fac).astype(np.int32)
        y_lim = -1 * (y_lim * y_fac).astype(np.int32)
        z_lim = 1 * (z_lim * z_fac).astype(np.int32)

        x_lim2 = x_lim[:]
        y_lim2 = y_lim[:]
        z_lim2 = z_lim[:]
        i_lim2 = i_lim[:]

        x_lim = x_lim[(x_lim2>-self.size[0]) & (x_lim2<= 0) & (y_lim2>-self.size[1]) & (y_lim2 <= 0) & (z_lim2<self.size[2]) & (z_lim2 >= 0)]
        y_lim = y_lim[(x_lim2>-self.size[0]) & (x_lim2<= 0) & (y_lim2>-self.size[1]) & (y_lim2 <= 0) & (z_lim2<self.size[2]) & (z_lim2 >= 0)]
        z_lim = z_lim[(x_lim2>-self.size[0]) & (x_lim2<= 0) & (y_lim2>-self.size[1]) & (y_lim2 <= 0) & (z_lim2<self.size[2]) & (z_lim2 >= 0)]
        i_lim = i_lim[(x_lim2>-self.size[0]) & (x_lim2<= 0) & (y_lim2>-self.size[1]) & (y_lim2 <= 0) & (z_lim2<self.size[2]) & (z_lim2 >= 0)]
        
        
        d = dict()
        for i in range(len(x_lim)):
            if (x_lim[i], y_lim[i]) in d:
                d[(x_lim[i], y_lim[i])].append(i_lim[i])
            else:
                d[(x_lim[i], y_lim[i])] = [i_lim[i]]
        
        
        img = np.zeros([self.size[0], self.size[1], self.size[2]+1], dtype=np.float32)
        img[x_lim, y_lim, z_lim] = 255.

        if self.interpolate:
            img2 = img.copy()


            img2[1:self.size[0]-1, 1:self.size[1]-1, 0:self.size[2]-1] = (img[0:self.size[0]-2, 0:self.size[1]-2, 0:self.size[2]-1]+
                                            img[0:self.size[0]-2, 2:self.size[1], 0:self.size[2]-1]+
                                            img[0:self.size[0]-2, 1:self.size[1]-1, 0:self.size[2]-1]+
                                            img[1:self.size[0]-1, 2:self.size[1], 0:self.size[2]-1]+
                                            img[1:self.size[0]-1, 0:self.size[1]-2, 0:self.size[2]-1]+
                                            img[2:self.size[0], 0:self.size[1]-2, 0:self.size[2]-1]+
                                            img[2:self.size[0], 2:self.size[1], 0:self.size[2]-1]+
                                            img[2:self.size[0], 1:self.size[1]-1, 0:self.size[2]-1]) / 8.
            
           
            img = np.maximum(img2, img)

        for k in d:
            img[k[0], k[1], -1] = max(d[k])*255.
             
        img = img[:,:, ::-1]
        img = img / 255.

        if self.fliplr:
            img = np.fliplr(img) 

        return img
        
        # img = np.zeros([self.size[0], self.size[1], self.size[2]+1], dtype=np.float32)
        # # occupancy grid
        # img[x_lim, y_lim, z_lim] = 255.
        # img[x_lim, y_lim, -1] = i_lim * 255.
        # img = img[:,:, ::-1]
        # img = (img - 127.) / 127.
        # return img


    def in_range_points(self, points, x, y, z, x_range, y_range, z_range):
        """ 
        This function is imported from https://github.com/windowsub0406/KITTI_Tutorial/blob/master/Convert_Velo_2_Topview_detail.ipynb
        extract in-range points 
        """
        return points[np.logical_and.reduce((x >= x_range[0], x <= x_range[1], y >= y_range[0], \
                                            y <= y_range[1], z >= z_range[0], z <= z_range[1]))]




    # def rotateImage(self, image, angle):
    #     image_center = tuple(np.array(image.shape[1::-1]) / 2)
    #     rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    #     result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    #     return result