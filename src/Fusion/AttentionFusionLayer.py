import tensorflow as tf

from ops.ops import *



def AttentionFusionLayerFunc2(rgb_features, fv_lidar_features, train_fusion_fv_lidar, original_lidar_feats, scope, batch_size, k_lidar=500, k_rgb=1000, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):

        lidar_features_sigmoid = conv(original_lidar_feats, 1, kernel=1, stride=1, scope=scope+'conv_bev', reuse=reuse)
        # lidar_features_sigmoid = batch_norm(lidar_features_sigmoid, scope='bn_fusion_lidar')
        lidar_features_sigmoid = tf.math.sigmoid(lidar_features_sigmoid)
        # print('lidar_features_sigmoid', lidar_features_sigmoid)
        lidar_features_sigmoid = tf.reshape(lidar_features_sigmoid, (-1, original_lidar_feats.shape[1]*original_lidar_feats.shape[2]))
        # print('lidar_features_sigmoid', lidar_features_sigmoid)
        _, lidar_fuse_indices = tf.math.top_k(lidar_features_sigmoid, k=k_lidar)


        rgb_features_sigmoid = conv(rgb_features, 1, kernel=1, stride=1, scope=scope+'conv_rgb', reuse=reuse)
        # rgb_features_sigmoid = batch_norm(rgb_features_sigmoid, scope='bn_fusion_rgb')
        rgb_features_sigmoid = tf.math.sigmoid(rgb_features_sigmoid)
        # print('rgb_features_sigmoid', rgb_features_sigmoid)
        rgb_features_sigmoid = tf.reshape(rgb_features_sigmoid, (-1, rgb_features.shape[1]*rgb_features.shape[2]))
        # print('rgb_features_sigmoid', rgb_features_sigmoid)
        _, rgb_fuse_indices = tf.math.top_k(rgb_features_sigmoid, k=k_rgb)

        # print('lidar_fuse_indices', lidar_fuse_indices)
        # print('rgb_fuse_indices', rgb_fuse_indices)

        original_lidar_feats_flattened = tf.reshape(original_lidar_feats, (-1, original_lidar_feats.shape[1]*original_lidar_feats.shape[2], original_lidar_feats.shape[-1]))
        rgb_feats_flattened = tf.reshape(rgb_features, (-1, rgb_features.shape[1]*rgb_features.shape[2], rgb_features.shape[-1]))

        # print('original_lidar_feats_flattened', original_lidar_feats_flattened)
        # print('rgb_feats_flattened', rgb_feats_flattened)

        # original_lidar_feats_flattened = tf.transpose(original_lidar_feats_flattened, (1, 0, 2))
        # rgb_feats_flattened = tf.transpose(rgb_feats_flattened, (1, 0, 2))

        # # print('original_lidar_feats_flattened', original_lidar_feats_flattened)
        # # print('rgb_feats_flattened', rgb_feats_flattened)

        lidar_to_fuse_feats = tf.batch_gather(original_lidar_feats_flattened, lidar_fuse_indices)
        rgb_to_fuse_feats = tf.batch_gather(rgb_feats_flattened, rgb_fuse_indices)

        # print('lidar_to_fuse_feats', lidar_to_fuse_feats)
        # print('rgb_to_fuse_feats', rgb_to_fuse_feats)

        # lidar_to_fuse_feats = tf.transpose(lidar_to_fuse_feats, (1, 0, 2))
        # rgb_to_fuse_feats = tf.transpose(rgb_to_fuse_feats, (1, 0, 2))

        # # print('lidar_to_fuse_feats', lidar_to_fuse_feats)
        # # print('rgb_to_fuse_feats', rgb_to_fuse_feats)

        _, A, _ = lidar_to_fuse_feats.shape
        _, X, _ = rgb_to_fuse_feats.shape

        X = int(X)
        A = int(A)

        Wce = tf.get_variable('Wce', [A, X, 1], initializer=tf.contrib.layers.xavier_initializer())
        Whe = tf.get_variable('Whe', [1, A, 1], initializer=tf.contrib.layers.xavier_initializer())
        Whe_2 = tf.get_variable('Whe_2', [1, A, 1, 1], initializer=tf.contrib.layers.xavier_initializer())

        # print('Wce', Wce)
        # print('Whe', Whe)
        # print('Whe_2', Whe_2)

        rgb_to_fuse_feats2 = tf.expand_dims(rgb_to_fuse_feats, 1)
        # print('rgb_to_fuse_feats2', rgb_to_fuse_feats2)
        lidar_to_fuse_feats2 = tf.expand_dims(lidar_to_fuse_feats, 2)
        # print('lidar_to_fuse_feats2', lidar_to_fuse_feats2)
        temp1 = tf.multiply(Whe, lidar_to_fuse_feats)
        # print('temp1', temp1)
        temp2 = tf.multiply(Wce, rgb_to_fuse_feats2)
        # print('temp2', temp2)
        temp1 = tf.expand_dims(temp1, 2)
        # print('temp1', temp1)
        et = Whe_2 * tf.tanh(temp1 + temp2)
        # print('et', et)
        at = tf.nn.softmax(et, axis=2)
        # print('at', at)
        rt = tf.multiply(lidar_to_fuse_feats2, tf.multiply(at, rgb_to_fuse_feats2))
        # print('rt', rt)
        rt = tf.reduce_sum(rt, axis=[2], keepdims=True)
        # print('rt', rt)
        rt = tf.squeeze(rt, [2])
        # print('rt', rt)
        # print('----------------------------------------')

        Wce2 = tf.get_variable('Wce2', [X, 1, 1], initializer=tf.contrib.layers.xavier_initializer())
        Whe2 = tf.get_variable('Whe2', [X, A, 1], initializer=tf.contrib.layers.xavier_initializer())
        Whe_22 = tf.get_variable('Whe_22', [1, X, 1, 1], initializer=tf.contrib.layers.xavier_initializer())

        # print('Wce2', Wce2)
        # print('Whe2', Whe2)
        # print('Whe_22', Whe_22)


        rgb_to_fuse_feats2 = tf.expand_dims(rgb_to_fuse_feats, 2)
        # print('rgb_to_fuse_feats2', rgb_to_fuse_feats2)
        lidar_to_fuse_feats2 = tf.expand_dims(lidar_to_fuse_feats, 1)
        # print('lidar_to_fuse_feats2', lidar_to_fuse_feats2)

        temp12 = tf.multiply(Whe2, lidar_to_fuse_feats2)
        # print('temp12', temp12)
        temp22 = tf.multiply(Wce2, rgb_to_fuse_feats2)
        # print('temp22', temp22)
        et2 = Whe_22 * tf.tanh(temp12 + temp22)
        # print('et2', et2)
        at2 = tf.nn.softmax(et2, axis=2)
        # print('at2', at2)
        rt2 = tf.multiply(rgb_to_fuse_feats2, tf.multiply(at2, lidar_to_fuse_feats2))
        # print('rt2', rt2)
        rt2 = tf.reduce_sum(rt2, axis=[2], keepdims=True)
        # print('rt2', rt2)
        rt2 = tf.squeeze(rt2, [2])
        # print('rt2', rt2)
        
        # print('-----------------------------')
        # rt = tf.transpose(rt, (1, 0, 2))
        # print('rt', rt)
        # print('lidar_fuse_indices', lidar_fuse_indices)
        # print('original_lidar_feats_flattened', original_lidar_feats_flattened)

        
        ref_lidar = tf.Variable(tf.zeros((batch_size, original_lidar_feats_flattened.shape[1], original_lidar_feats_flattened.shape[2])),
                      name="ref_lidar", validate_shape=False, dtype=tf.float32)
        ref_rgb = tf.Variable(tf.zeros((batch_size, rgb_feats_flattened.shape[1], rgb_feats_flattened.shape[2])),
                      name="ref_lidar", validate_shape=False, dtype=tf.float32)
        
        lidar_fused_feats = tf.batch_scatter_update(ref_lidar, lidar_fuse_indices, rt)
        rgb_fused_feats = tf.batch_scatter_update(ref_rgb, rgb_fuse_indices, rt2)
        # print('lidar_fused_feats', lidar_fused_feats)
        # print('rgb_fused_feats', rgb_fused_feats)

        lidar_fused_feats = tf.reshape(lidar_fused_feats, tf.shape(original_lidar_feats))
        rgb_fused_feats = tf.reshape(rgb_fused_feats, tf.shape(rgb_features))

        lidar_fused_feats = lidar_fused_feats + original_lidar_feats
        rgb_fused_feats = rgb_fused_feats + rgb_features

        # print('lidar_fused_feats', lidar_fused_feats)
        # print('rgb_fused_feats', rgb_fused_feats)


    return lidar_fused_feats, rgb_fused_feats





def AttentionFusionLayerFunc(rgb_features, fv_lidar_features, train_fusion_fv_lidar, original_lidar_feats, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        # rgb_features = conv(rgb_features, 64, kernel=1, stride=1, scope=scope+'conv_rgb', reuse=reuse)
        # rgb_features = relu(rgb_features)

        lidar_features = conv(original_lidar_feats, 192, kernel=1, stride=1, scope=scope+'conv_bev', reuse=reuse)
        lidar_features = batch_norm(lidar_features, scope='bn_fusion_lidar')
        lidar_features = relu(lidar_features)

        _, A, B, _ = rgb_features.shape
        _, X, Y, _ = lidar_features.shape
        X = int(X)
        Y = int(Y)
        A = int(A)
        B = int(B)
        Wce = tf.get_variable('Wce', [X, Y, A, B, 1], initializer=tf.contrib.layers.xavier_initializer())
        Whe = tf.get_variable('Whe', [X, Y, 1, 1], initializer=tf.contrib.layers.xavier_initializer())
        Whe_2 = tf.get_variable('Whe_2', [1, X, Y, 1, 1], initializer=tf.contrib.layers.xavier_initializer())

        lidar_features = tf.expand_dims(lidar_features, -2)

        rgb_features = tf.expand_dims(rgb_features, 1)
        rgb_features = tf.expand_dims(rgb_features, 1)

        # # print('rgb_features', rgb_features)
        # # print('Wce', Wce)

        temp1 = tf.multiply(Whe, lidar_features)
        temp1 = tf.expand_dims(temp1, -2)

        temp2 = tf.multiply(Wce, rgb_features)
        # # print('temp2', temp2)
       

        # # print('temp1', temp1)
        # # print('temp2', temp2)

        et = temp1 + temp2
        # # print('et', et)
        et = tf.reshape(et, (-1, X, Y, int(A * B), lidar_features.get_shape()[-1]))
        # # print('et', et)
        at = tf.nn.softmax(et, axis=3)

        temp1 = tf.multiply(Whe_2, lidar_features)
        # # print('tmp1', temp1)
        temp1 = tf.nn.sigmoid(temp1) ## check the axis
        # # print('tmp1', temp1)
        temp1 = tf.reshape(temp1, (-1, X, Y, 1, 1, lidar_features.get_shape()[-1]))

        at = tf.reshape(at, (-1, X, Y, A, B, lidar_features.get_shape()[-1]))
        # # print('at', at)
        # # print('rgb_features', rgb_features)
        rt = tf.multiply(temp1, tf.multiply(at, rgb_features))
        # # print('rrrrttttt', rt)
        rt = tf.reduce_sum(rt, axis=[3, 4], keepdims=True)
        # # print('rrrrttttt', rt)
        rt = tf.squeeze(rt, [3, 4])

    if fv_lidar_features is not None:
        with tf.variable_scope('fv_fusion'):
    
                rt_with_fv = tf.reduce_sum(tf.multiply(temp1, tf.multiply(at, fv_lidar_features)), axis=[3, 4], keepdims=True)
                rt_with_fv = tf.squeeze(rt_with_fv, [3, 4])

                rt_with_fv = tf.concat([rt, rt_with_fv], axis=-1)
                
                rt_with_fv = conv(rt_with_fv, 192, kernel=1, stride=1, scope='atention_fusion_3_adjust_channels_conv', reuse=reuse)
                rt_with_fv = relu(rt_with_fv)
                
        with tf.variable_scope(scope, reuse=reuse):
            rt = tf.cond(train_fusion_fv_lidar, lambda: rt_with_fv, lambda: rt, name='fv_fusion_cond')

    with tf.variable_scope(scope, reuse=reuse):
        original_lidar_feats = tf.concat([rt, original_lidar_feats], axis=-1)
        original_lidar_feats = conv(original_lidar_feats, 256, kernel=1, stride=1, scope='final_conv_between_modalities', reuse=reuse)
        original_lidar_feats = batch_norm(original_lidar_feats, scope='final_bn_between_modalities')
        original_lidar_feats = relu(original_lidar_feats)

    return original_lidar_feats





# def AttentionFusionLayerFunc(rgb_features, fv_lidar_features, train_fusion_fv_lidar, lidar_features, scope, reuse=False):
#     with tf.variable_scope(scope, reuse=reuse):
#         rgb_features = conv(rgb_features, 64, kernel=1, stride=1, scope=scope+'conv_rgb', reuse=reuse)
#         rgb_features = relu(rgb_features)

#         lidar_features = conv(lidar_features, 64, kernel=1, stride=1, scope=scope+'conv_bev', reuse=reuse)
#         lidar_features = relu(lidar_features)

#         _, A, B, _ = rgb_features.shape
#         _, X, Y, _ = lidar_features.shape
#         X = int(X)
#         Y = int(Y)
#         A = int(A)
#         B = int(B)
#         Wce = tf.get_variable('Wce', [A, B, 1, 1], initializer=tf.contrib.layers.xavier_initializer())
#         Whe = tf.get_variable('Whe', [X, Y, 1, 1], initializer=tf.contrib.layers.xavier_initializer())
#         Whe_2 = tf.get_variable('Whe_2', [1, X, Y, 1, 1], initializer=tf.contrib.layers.xavier_initializer())

#         lidar_features = tf.expand_dims(lidar_features, -2)

#         rgb_features = tf.expand_dims(rgb_features, 1)
#         rgb_features = tf.expand_dims(rgb_features, -2)

#         temp1 = tf.multiply(Whe, lidar_features)
#         temp1 = tf.expand_dims(temp1, -2)
#         temp2 = tf.multiply(Wce, rgb_features)
#         temp2 = tf.squeeze(temp2, -2)
#         temp2 = tf.expand_dims(temp2, 1)

#         # # # print('temp1', temp1)
#         # # # print('temp2', temp2)

#         et = tf.nn.tanh(tf.multiply(temp1, temp2))
#         # # # print('et', et)
#         et = tf.reshape(et, (-1, X, Y, int(A * B), lidar_features.get_shape()[-1]))
#         # # # print('et', et)
#         at = tf.nn.softmax(et, axis=3)

#         temp1 = tf.multiply(Whe_2, lidar_features)
#         temp1 = tf.nn.sigmoid(temp1) ## check the axis
#         temp1 = tf.reshape(temp1, (-1, X, Y, 1, 1, lidar_features.get_shape()[-1]))

#         at = tf.reshape(at, (-1, X, Y, A, B, lidar_features.get_shape()[-1]))

#         rgb_features = tf.expand_dims(rgb_features, 1)
#         rgb_features = tf.squeeze(rgb_features, -2)

#         rt = tf.reduce_sum(tf.multiply(temp1, tf.multiply(at, rgb_features)), axis=[3, 4], keepdims=True)
#         rt = tf.squeeze(rt, [3, 4])

#     if fv_lidar_features is not None:
#         with tf.variable_scope('fv_fusion'):
    
#                 rt_with_fv = tf.reduce_sum(tf.multiply(temp1, tf.multiply(at, fv_lidar_features)), axis=[3, 4], keepdims=True)
#                 rt_with_fv = tf.squeeze(rt_with_fv, [3, 4])

#                 rt_with_fv = tf.concat([rt, rt_with_fv], axis=-1)
                
#                 rt_with_fv = conv(rt_with_fv, 64, kernel=1, stride=1, scope='atention_fusion_3_adjust_channels_conv', reuse=reuse)
#                 rt_with_fv = relu(rt_with_fv)
                
#         with tf.variable_scope(scope, reuse=reuse):
#             rt = tf.cond(train_fusion_fv_lidar, lambda: rt_with_fv, lambda: rt, name='fv_fusion_cond')

#     return rt
