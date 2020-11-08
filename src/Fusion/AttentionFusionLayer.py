import tensorflow as tf

from ops.ops import *


def AttentionFusionLayerFunc3(original_rgb_features, fv_lidar_features, train_fusion_fv_lidar, original_lidar_feats, scope, is_training=True, kernel_lidar=5, kernel_rgb=5, stride_lidar=3, stride_rgb=3, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        
        lidar_features = conv(original_lidar_feats, 64, kernel=kernel_lidar, stride=stride_lidar, scope=scope+'conv_bev', reuse=reuse)
        lidar_features = batch_norm(lidar_features, is_training=is_training, scope='bn_fusion_lidar')
        lidar_features = relu(lidar_features)
        lidar_features = dropout(lidar_features, rate=0.2, scope='dropout_lidar_feats', is_training=is_training)

        # print('lidar_features', lidar_features)

        rgb_features = conv(original_rgb_features, 64,  kernel=kernel_rgb, stride=stride_rgb, scope=scope+'conv_rgb', reuse=reuse)
        rgb_features = batch_norm(rgb_features, is_training=is_training, scope='bn_fusion_rgb')
        rgb_features = relu(rgb_features)
        rgb_features = dropout(rgb_features, rate=0.2, scope='dropout_lidar_feats', is_training=is_training)

        # print('rgb_features', rgb_features)

        _, A, B, _ = rgb_features.shape
        _, X, Y, _ = lidar_features.shape
        X = int(X)
        Y = int(Y)
        A = int(A)
        B = int(B)
        Wce = tf.get_variable('Wce', [X, Y, A, B, 1], initializer=tf.contrib.layers.xavier_initializer())
        Whe = tf.get_variable('Whe', [X, Y, 1, 1], initializer=tf.contrib.layers.xavier_initializer())
        Whe_2 = tf.get_variable('Whe_2', [1, X, Y, 1, 1], initializer=tf.contrib.layers.xavier_initializer())

        Wce2 = tf.get_variable('Wce2', [A, B, X, Y, 1], initializer=tf.contrib.layers.xavier_initializer())
        Whe2 = tf.get_variable('Whe2', [A, B, 1, 1], initializer=tf.contrib.layers.xavier_initializer())
        Whe_22 = tf.get_variable('Whe_22', [1, A, B, 1, 1], initializer=tf.contrib.layers.xavier_initializer())


        lidar_features2 = tf.expand_dims(lidar_features, -2)

        rgb_features2 = tf.expand_dims(rgb_features, 1)
        rgb_features2 = tf.expand_dims(rgb_features2, 1)

        temp1 = tf.multiply(Whe, lidar_features2)
        temp1 = tf.expand_dims(temp1, -2)
        temp2 = tf.multiply(Wce, rgb_features2)       

        et = temp1 + temp2
        et = tf.reshape(et, (-1, X, Y, int(A * B), lidar_features2.get_shape()[-1]))
        at = tf.nn.softmax(et, axis=3)

        temp1 = tf.multiply(Whe_2, lidar_features2)
        temp1 = tf.nn.sigmoid(temp1) ## check the axis
        temp1 = tf.reshape(temp1, (-1, X, Y, 1, 1, lidar_features2.get_shape()[-1]))

        at = tf.reshape(at, (-1, X, Y, A, B, lidar_features2.get_shape()[-1]))
        rt = tf.multiply(temp1, tf.multiply(at, rgb_features2))
        rt = tf.reduce_sum(rt, axis=[3, 4], keepdims=True)
        rt = tf.squeeze(rt, [3, 4])


        ###

        lidar_features2 = tf.expand_dims(lidar_features, 1)
        lidar_features2 = tf.expand_dims(lidar_features2, 1)

        rgb_features2 = tf.expand_dims(rgb_features, -2)

        temp1 = tf.multiply(Whe2, rgb_features2)
        temp1 = tf.expand_dims(temp1, -2)
        temp2 = tf.multiply(Wce2, lidar_features2)       

        et = temp1 + temp2
        et = tf.reshape(et, (-1, A, B, int(X * Y), rgb_features2.get_shape()[-1]))
        at = tf.nn.softmax(et, axis=3)

        temp1 = tf.multiply(Whe_22, rgb_features2)
        temp1 = tf.nn.sigmoid(temp1) ## check the axis
        temp1 = tf.reshape(temp1, (-1, A, B, 1, 1, rgb_features2.get_shape()[-1]))

        at = tf.reshape(at, (-1, A, B, X, Y, rgb_features2.get_shape()[-1]))
        rt2 = tf.multiply(temp1, tf.multiply(at, lidar_features2))
        rt2 = tf.reduce_sum(rt2, axis=[3, 4], keepdims=True)
        rt2 = tf.squeeze(rt2, [3, 4])


        ###

        # print('rt', rt)
        # print('rt2', rt2)
        # print('original_rgb_features', original_rgb_features)
        # print('original_lidar_feats', original_lidar_feats)

        rt = upsample(rt, is_training=is_training, size=(stride_lidar, stride_lidar), scope='rt_upsample', use_deconv=True, kernel_size=6)
        rt2 = upsample(rt2, is_training=is_training, size=(stride_rgb, stride_rgb), scope='rt2_upsample', use_deconv=True, kernel_size=6)

        # print('rt', rt)
        # print('rt2', rt2)


        new_layer_height = rt.get_shape()[1]
        prev_layer_height = original_lidar_feats.get_shape()[1]
        diff = new_layer_height - prev_layer_height
        half_diff = diff//2

        new_layer_width = rt.get_shape()[2]
        prev_layer_width = original_lidar_feats.get_shape()[2]
        diff_w = new_layer_width - prev_layer_width
        half_diff_w = diff_w//2
    
        rt = crop(rt, ((int(half_diff), int(diff-half_diff)), (int(half_diff_w), int(diff_w-half_diff_w))), 'crop_rt')


        new_layer_height = rt2.get_shape()[1]
        prev_layer_height = original_rgb_features.get_shape()[1]
        diff = new_layer_height - prev_layer_height
        half_diff = diff//2

        new_layer_width = rt2.get_shape()[2]
        prev_layer_width = original_rgb_features.get_shape()[2]
        diff_w = new_layer_width - prev_layer_width
        half_diff_w = diff_w//2
    
        rt2 = crop(rt2, ((int(half_diff), int(diff-half_diff)), (int(half_diff_w), int(diff_w-half_diff_w))), 'crop_rt2')

        # print('rt', rt)
        # print('rt2', rt2)
        
        c1 = int(original_lidar_feats.get_shape()[-1])
        original_lidar_feats2 = tf.concat([rt, original_lidar_feats], axis=-1)
        original_lidar_feats2 = conv(original_lidar_feats2, c1, kernel=1, stride=1, scope='final_conv_between_modalities_lidar', reuse=reuse)
        original_lidar_feats2 = batch_norm(original_lidar_feats2, is_training=is_training, scope='final_bn_between_modalities_lidar')
        original_lidar_feats2 = relu(original_lidar_feats2)
        # original_lidar_feats2 = dropout(original_lidar_feats2, rate=0.5, scope='dropout_lidar')

        c2 = int(original_rgb_features.get_shape()[-1])
        original_rgb_features2 = tf.concat([rt2, original_rgb_features], axis=-1)
        original_rgb_features2 = conv(original_rgb_features2, c2, kernel=1, stride=1, scope='final_conv_between_modalities_rgb', reuse=reuse)
        original_rgb_features2 = batch_norm(original_rgb_features2, is_training=is_training, scope='final_bn_between_modalities_rgb')
        original_rgb_features2 = relu(original_rgb_features2)
        # original_rgb_features2 = dropout(original_rgb_features2, rate=0.5, scope='dropout_rgb')
        

    return original_lidar_feats2, original_rgb_features2


