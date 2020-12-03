
import tensorflow as tf
import numpy as np

def generate_corners(truth2, predictions2):

        truth = truth2[:, :, :, :, :8]
        predictions3 = predictions2[:, :, :, :, :8]

        mins = np.array([-0.5, -0.5, 0, 0.8, 0.3, 0.13, -1.1, -1.1])
        maxs = np.array([0.5, 0.5, 1, 2.6, 1.4, 0.82, 1.1, 1.1])
        mins = np.expand_dims(mins, [0, 1, 2])
        maxs = np.expand_dims(maxs, [0, 1, 2])
        
        
        truth3 = ((truth + 1) / 2) * (maxs - mins) + mins 
        predictions4 = ((tf.nn.tanh(predictions3) + 1) / 2) * (maxs - mins) + mins 

        size_true = tf.exp(truth3[:, :, :, :, 3:6])
        size_pred = tf.exp(predictions4[:, :, :, :, 3:6])

        predictions = tf.math.sigmoid(predictions3)-0.5

        theta_pred = (tf.math.sigmoid(predictions[:, :, :, :, 6]) * np.pi/2) 
        theta_truth = (truth[:, :, :, :, 6] + np.pi/4) 

       
       
        dx = predictions[:, :, :, :, 0]
        dy = predictions[:, :, :, :, 1]
        cos_t = tf.math.cos(theta_pred)
        sin_t = tf.math.sin(theta_pred)
        l = size_pred[:, :, :, :, 0]
        w = size_pred[:, :, :, :, 1]

        x = tf.range(0, 70, 0.625)
        y = tf.range(-40, 40, 0.625)
        yy, xx = tf.meshgrid(y, x)
        yy = tf.expand_dims(yy, axis=-1)
        xx = tf.expand_dims(xx, axis=-1)
        centre_y = yy + dy
        centre_x = xx + dx
       
       
        rear_left_x = centre_x - l/2 * cos_t - w/2 * sin_t
        rear_left_x = tf.expand_dims(rear_left_x, axis=-1)
        rear_left_y = centre_y - l/2 * sin_t + w/2 * cos_t
        rear_left_y = tf.expand_dims(rear_left_y, axis=-1)
        rear_right_x = centre_x - l/2 * cos_t + w/2 * sin_t
        rear_right_x = tf.expand_dims(rear_right_x, axis=-1)
        rear_right_y = centre_y - l/2 * sin_t - w/2 * cos_t
        rear_right_y = tf.expand_dims(rear_right_y, axis=-1)
        front_right_x = centre_x + l/2 * cos_t + w/2 * sin_t
        front_right_x = tf.expand_dims(front_right_x, axis=-1)
        front_right_y = centre_y + l/2 * sin_t - w/2 * cos_t
        front_right_y = tf.expand_dims(front_right_y, axis=-1)
        front_left_x = centre_x + l/2 * cos_t - w/2 * sin_t
        front_left_x = tf.expand_dims(front_left_x, axis=-1)
        front_left_y = centre_y + l/2 * sin_t + w/2 * cos_t
        front_left_y = tf.expand_dims(front_left_y, axis=-1)

        decoded_reg_pred = tf.concat([rear_left_x, rear_left_y, rear_right_x, rear_right_y,
                                 front_right_x, front_right_y, front_left_x, front_left_y], axis=-1)


        dx = truth[:, :, :, :, 0]
        dy = truth[:, :, :, :, 1]
        cos_t = tf.math.cos(theta_truth)
        sin_t = tf.math.sin(theta_truth)
        l = size_true[:, :, :, :, 0]
        w = size_true[:, :, :, :, 1]

        x = tf.range(0, 70, 0.625)
        y = tf.range(-40, 40, 0.625)
        yy, xx = tf.meshgrid(y, x)
        yy = tf.expand_dims(yy, axis=-1)
        xx = tf.expand_dims(xx, axis=-1)
        centre_y = yy + dy
        centre_x = xx + dx
       
       
        rear_left_x = centre_x - l/2 * cos_t - w/2 * sin_t
        rear_left_x = tf.expand_dims(rear_left_x, axis=-1)
        rear_left_y = centre_y - l/2 * sin_t + w/2 * cos_t
        rear_left_y = tf.expand_dims(rear_left_y, axis=-1)
        rear_right_x = centre_x - l/2 * cos_t + w/2 * sin_t
        rear_right_x = tf.expand_dims(rear_right_x, axis=-1)
        rear_right_y = centre_y - l/2 * sin_t - w/2 * cos_t
        rear_right_y = tf.expand_dims(rear_right_y, axis=-1)
        front_right_x = centre_x + l/2 * cos_t + w/2 * sin_t
        front_right_x = tf.expand_dims(front_right_x, axis=-1)
        front_right_y = centre_y + l/2 * sin_t - w/2 * cos_t
        front_right_y = tf.expand_dims(front_right_y, axis=-1)
        front_left_x = centre_x + l/2 * cos_t - w/2 * sin_t
        front_left_x = tf.expand_dims(front_left_x, axis=-1)
        front_left_y = centre_y + l/2 * sin_t + w/2 * cos_t
        front_left_y = tf.expand_dims(front_left_y, axis=-1)

        decoded_reg_truth = tf.concat([rear_left_x, rear_left_y, rear_right_x, rear_right_y,
                                 front_right_x, front_right_y, front_left_x, front_left_y], axis=-1)
        print('decoded_reg_truth', decoded_reg_truth)
        return decoded_reg_truth, decoded_reg_pred


def generate_corners_height(truth2, predictions2):

        truth = truth2[:, :, :, :, :8]
        predictions3 = predictions2[:, :, :, :, :8]

        mins = np.array([-0.5, -0.5, 0, 0.8, 0.3, 0.13, -1.1, -1.1])
        maxs = np.array([0.5, 0.5, 1, 2.6, 1.4, 0.82, 1.1, 1.1])
        mins = np.expand_dims(mins, [0, 1, 2])
        maxs = np.expand_dims(maxs, [0, 1, 2])
        
        
        truth3 = ((truth + 1) / 2) * (maxs - mins) + mins 
        predictions4 = ((tf.nn.tanh(predictions3) + 1) / 2) * (maxs - mins) + mins 

        size_true = tf.exp(truth3[:, :, :, :, 3:6])
        size_pred = tf.exp(predictions4[:, :, :, :, 3:6])

        predictions = tf.math.sigmoid(predictions3)-0.5

        theta_pred = (tf.math.sigmoid(predictions[:, :, :, :, 6]) * np.pi/2) 
        theta_truth = (truth[:, :, :, :, 6] + np.pi/4) 

       
       
        dx = predictions[:, :, :, :, 2]

        l = size_pred[:, :, :, :, 2]
        # l = tf.expand_dims(l, -1)

        x = np.arange(-2.5, 1, 0.1)
        x = np.expand_dims(x, 0)
        x = np.expand_dims(x, 1)
        x = np.repeat(x, 112, 1)
        x = np.expand_dims(x, 2)
        x = np.repeat(x, 128, 2)
        x = np.expand_dims(x, 3)
        x = np.repeat(x, 2, 3)
        # x = np.expand_dims(x, 4)
        # dx = tf.expand_dims(dx, -1)
        # centre_y = x + dx
        centre_y = 0.5 + dx

        down_z = centre_y - l/2 
        down_z = tf.expand_dims(down_z, axis=-1)
        
        up_z = centre_y + l/2
        up_z = tf.expand_dims(up_z, axis=-1)
       

        decoded_reg_pred = tf.concat([down_z, up_z], axis=-1)


        dx = truth[:, :, :, :, 2]
        dy = truth[:, :, :, :, 1]
        cos_t = 1
        sin_t = 0
        l = size_true[:, :, :, :, 2]
        # l = tf.expand_dims(l, -1)

        x = np.arange(-2.5, 1, 0.1)
        x = np.expand_dims(x, 0)
        x = np.expand_dims(x, 1)
        x = np.repeat(x, 112, 1)
        x = np.expand_dims(x, 2)
        x = np.repeat(x, 128, 2)
        x = np.expand_dims(x, 3)
        x = np.repeat(x, 2, 3)
        # dx = tf.expand_dims(dx, -1)
        # centre_y = x + dx
        centre_y = 0.5 + dx

        down_z = centre_y - l/2 
        down_z = tf.expand_dims(down_z, axis=-1)
        
        up_z = centre_y + l/2
        up_z = tf.expand_dims(up_z, axis=-1)
       

        decoded_reg_truth = tf.concat([down_z, up_z], axis=-1)

        return decoded_reg_truth, decoded_reg_pred


def get_corners_loss(truth, predictions):
    decoded_reg_truth, decoded_reg_pred = generate_corners(truth, predictions)
    loss_bev = tf.math.reduce_mean(tf.math.squared_difference(decoded_reg_truth, decoded_reg_pred), axis=-1)
    # loss_height = tf.math.reduce_mean(tf.math.squared_difference(decoded_reg_truth_height, decoded_reg_truth_height), axis=-1)
    return loss_bev


def get_corners_loss_height(truth, predictions):
    decoded_reg_truth_height, decoded_reg_truth_height = generate_corners_height(truth, predictions)
    print('decoded_reg_truth_height', decoded_reg_truth_height)
    loss_height = tf.math.reduce_mean(tf.math.squared_difference(decoded_reg_truth_height, decoded_reg_truth_height), axis=-1)
    print('loss_height ', loss_height)
    loss_height = tf.math.reduce_sum(loss_height, axis=1)
    loss_height = tf.expand_dims(loss_height, axis=1)
    return loss_height


