import numpy as np
import cv2
from data.data_utils.velodyne_points import *
import math
from utils.utils import *
import os
import tensorflow as tf
import matplotlib.patches as patches


from data.data_utils.fv_utils import *



############################
########IMAGES##############
############################

def __read_camera(image_path, image_size, translate_x = 0, translate_y = 0, fliplr=False):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (3, image_size[0]), interpolation = cv2.INTER_AREA)
    if translate_x > 0:
        image [:, abs(translate_x):image_size[1]]= image[:, :image_size[1]-abs(translate_x)]
        image[:, :abs(translate_x)] = 0
    elif translate_x < 0:
        image [:, :image_size[1]-abs(translate_x)]= image[:, abs(translate_x):image_size[1]]
        image[:, image_size[1]-abs(translate_x):] = 0

    if translate_y > 0:
        image [abs(translate_y):image_size[0], :]= image[:image_size[0]-abs(translate_y), :]
        image[:abs(translate_y), :] = 0
    elif translate_y < 0:
        image [:image_size[0]-abs(translate_y), :]= image[abs(translate_y):image_size[0], :]
        image[image_size[0]-abs(translate_y):, :] = 0
    
    if fliplr:
        image = np.fliplr(image)
    return image, 0, 0

def read_camera(image_path, image_size, translate_x = 0, translate_y = 0, fliplr=False):
    return __read_camera(image_path, image_size, translate_x = translate_x, translate_y = translate_x, fliplr=fliplr)


############################
########LIDAR###############
############################

def read_lidar(rot, tr, sc, lidar_path, calib_path, lidar_size, img_width=1224, img_height=370, translate_x=0, translate_y=0, translate_z=0, ang=0, fliplr=False):
    image = velo_points_bev(rot, tr, sc, lidar_path, calib_path, size=lidar_size, img_width=img_width, img_height=img_height,
                     translate_x=translate_x, translate_y=translate_y, translate_z=translate_z, ang=ang, fliplr=fliplr)
    
    return image


def read_pc_fv(calib_path, velo_path):
    calib = Calibration(calib_path)
    pc_velo = load_velo_scan(velo_path)
    velo_fv = show_lidar_on_image(pc_velo[:, :3], calib, 1224, 370, 64)
    return velo_fv


############################
########CALIB###############
############################

def read_calib(calib_path):
    lines = []
    with open(calib_path) as calib_file:
        lines = calib_file.readlines()
    lines = list(filter(lambda x: len(x.split()) > 0, lines))
    calib_data = dict(list(map(lambda x: (x.split()[0][:-1], np.array(list(map(float, x.split()[1:])))), lines)))
    return calib_data






############################
########LABELS##############
############################



def cart2hom(pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom

def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

def project_rect_to_ref(pts_3d_rect, R0):
        ''' Input and Output are nx3 points '''
        # return np.transpose(np.dot(np.linalg.inv(RO), np.transpose(pts_3d_rect)))
        return np.transpose(np.dot(np.linalg.inv(R0.reshape((3, 3))), np.transpose(pts_3d_rect)))
    
# def project_ref_to_velo(pts_3d_ref, Tr_velo_to_cam):
#         C2V = inverse_rigid_trans(Tr_velo_to_cam)
#         pts_3d_ref = cart2hom(pts_3d_ref) # nx4
#         return np.dot(pts_3d_ref, np.transpose(C2V))

def project_ref_to_velo(pts_3d_ref, Tr_velo_to_cam):
        pts_3d_ref = cart2hom(pts_3d_ref) # nx4
        C2V = inverse_rigid_trans(Tr_velo_to_cam.reshape((3, 4)))
        return np.dot(pts_3d_ref, np.transpose(C2V))
    
def project_rect_to_velo2(rot, tr, sc, pts_3d_rect, RO, Tr_velo_to_cam):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        ''' 
        pts_3d_ref = project_rect_to_ref(pts_3d_rect, RO)
        temp = project_ref_to_velo(pts_3d_ref, Tr_velo_to_cam)
        temp = temp.transpose() + tr[:3, :1]
        temp = np.dot(sc[:3, :3], np.dot(rot[:3, :3], temp)).transpose()
        return temp

def project_point_from_camera_coor_to_velo_coor2(rot, tr, sc, location, dimemsion, agnle, calib_data):
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
    temp = result
    temp = temp.transpose() + tr[:3, :1]
    temp = np.dot(sc[:3, :3], np.dot(rot[:3, :3], temp)).transpose()
    return temp
    

def read_label(rot, tr, sc, label_path, calib_path, shift_h, shift_w, x_range=(0, 71), y_range=(-40, 40), z_range=(-3.0, 1), 
                    size=(512, 448, 40), get_actual_dims=False, from_file=True, translate_x=0, translate_y=0, translate_z=0, ang=0, get_neg=False, fliplr=False):

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
            # lines = list(filter(lambda x: len(x) > 0 and ( x[0] not in ['Car', 'Van', 'Truck', 'Tram', 'DontCare']), lines))
            lines = list(filter(lambda x: len(x) > 0 and ( x[0] not in ['Car']), lines))
            if len(lines) > 0:
                lines = lines[:1]
        else:
            # lines = list(filter(lambda x: len(x) > 0 and ( x[0] in ['Car', 'Van', 'Truck', 'Tram']), lines))
             lines = list(filter(lambda x: len(x) > 0 and ( x[0] in ['Car']), lines))
    
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
    print(locations.shape)
    if len(locations) > 0 and len(locations[0]) > 0:
        locations = project_rect_to_velo2(rot, tr, sc, locations, calib_data['R0_rect'].reshape((3, 3)), calib_data['Tr_velo_to_cam'].reshape((3, 4)))
    print(locations.shape)
    # print(z_range)

    indx = []
    i = 0
    for point in locations:
        if (point[0] >= x_range[0]  and point[0] <= x_range[1])\
            and (point[1] >= y_range[0] and point[1] <= y_range[1])\
            and (point[2] >= z_range[0] and point[2] <= z_range[1]):
            indx.append(i)
        i += 1

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

    points = [project_point_from_camera_coor_to_velo_coor2(rot, tr, sc, [location_x[i], location_y[i], location_z[i]], 
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
            dimension_height[i] = math.sqrt((-(b[0][2]+(-1*z_range[1]))*z_fac-(-b[4][2]+z_range[1])*z_fac)**2)

      
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
    if fliplr:
        for i in range(len(locations)):
            h = size[1]
            output[i][1] = h - output[i][1]

    if ang != 0:
        for i in range(len(locations)):
            # w = size[0]
            # h = size[1]
            # output[i][0], output[i][1] = rotate2((w//2, h//2), (output[i][0], output[i][1]), ang / 57.2958)
            output[i][6] = output[i][6] - ang / 57.2958

    output = list(filter(lambda point: 0 <= point[0] < size[0] and 0 <= point[1] < size[1] and 0 <= point[2] < size[2] , output))
    output = np.array(output)

    return points, output, calib_data['Tr_velo_to_cam'], calib_data['R0_rect'], calib_data['P2'], directions


def rotate(origin, point, angle):
#     https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy