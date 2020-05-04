import tensorflow as tf

from ops.ops import *
from utils.utils import *

class FusionLayer(tf.keras.layers.Layer):
    def __init__(self, dense_bev_height, dense_bev_width, 
                lidar_scale_height, lidar_scale_width, rgb_scale_height, rgb_scale_width,
                rgb_height, rgb_width,
                layer_nodes, kdtree, k, scope, Tr_velo_to_cam, R0_rect, P3, shift_h, shift_w,
                is_training=True, reuse=False, bev_ch=32):
        super(type(self), self).__init__()
        # self.image_feature_maps = image_feature_maps
        self.dense_bev_height = dense_bev_height
        self.dense_bev_width = dense_bev_width 
        self.lidar_scale_height = lidar_scale_height
        self.lidar_scale_width = lidar_scale_width
        self.rgb_scale_height = rgb_scale_height
        self.rgb_scale_width = rgb_scale_width
        self.rgb_height = rgb_height
        self.rgb_width = rgb_width
        self.layer_nodes = layer_nodes
        self.kdtree = kdtree
        self.k = k
        self.scope = scope
        self.Tr_velo_to_cam = Tr_velo_to_cam
        self.R0_rect = R0_rect
        self.P3 = P3 
        self.is_training = is_training
        self.reuse = reuse
        self.bev_ch = bev_ch
        self.shift_h = shift_h
        self.shift_w = shift_w
        
        
    def build(self, input_shape):
        # n_input = (int(input_shape[2]) + 3)*self.k
        n_input = (256+ 3)*self.k
        n_hidden1 = self.layer_nodes[0]
        n_hidden2 = self.layer_nodes[1]
        n_output = self.layer_nodes[2]


        self.b1 = self.add_variable('b1', [n_hidden1], initializer=tf.contrib.layers.xavier_initializer())
        self.b2 = self.add_variable('b2',[n_hidden2] , initializer=tf.contrib.layers.xavier_initializer())
        self.b3 = self.add_variable('b3', [n_output], initializer=tf.contrib.layers.xavier_initializer())

        self.w1 = self.add_variable('w1', [n_input, n_hidden1],  initializer=tf.contrib.layers.xavier_initializer())
        self.w2 = self.add_variable('w2', [n_hidden1, n_hidden2], initializer=tf.contrib.layers.xavier_initializer())
        self.w3 = self.add_variable('w3', [n_hidden2, n_output], initializer=tf.contrib.layers.xavier_initializer())


    # def compute_output_shape(self, input_shape):
    #     return (self.dense_bev_height, self.dense_bev_width, 256)

    def call(self, input):
        def MLP(input_feats):
            layer_1 = tf.nn.relu(tf.add(tf.matmul(input_feats, self.w1), self.b1))
            layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.w2), self.b2))
            out_layer = tf.add(tf.matmul(layer_2, self.w3), self.b3)
            return out_layer

        def fn_width(h, i, original_i, j, out):
            print('width')
            original_j = j * tf.cast(self.lidar_scale_width, tf.int32)
            k_points = self.kdtree[:, tf.cast(original_i, tf.int32), tf.cast(original_j, tf.int32), tf.cast(h, tf.int32), :, :]
            current_point = tf.constant(np.array([[1]]), dtype=tf.int32)
            current_point = tf.concat([current_point, [[tf.cast(original_i, tf.int32)]], [[tf.cast(original_j, tf.int32)]], [[tf.cast(h, tf.int32)]], [[tf.cast(1, tf.int32)]]], 0)
            current_point = tf.cast(tf.reshape(current_point[:4, :], [1, 4]), tf.float16)

            distances = tf.transpose(tf.map_fn(lambda k_points_batch: k_points_batch - current_point, tf.cast(k_points, tf.float16)), [1, 0, 2])
            rgb_points = tf.map_fn(lambda k_point_: project_point_from_lidar_to_image(k_point_, self.Tr_velo_to_cam, self.R0_rect, self.P3, self.shift_h, self.shift_w), 
                                        tf.cast(tf.transpose(k_points, [1, 0, 2]), tf.float32))
            rgb_points = tf.cast(rgb_points, tf.int32)
        
            
            def get_rgb_feats(rgb_point):
                pixel_features = tf.map_fn(lambda rgb_point_: input[:, tf.cast(rgb_point_[0] / self.rgb_scale_height, tf.int32), tf.cast(rgb_point_[1]/self.rgb_scale_width, tf.int32), :], rgb_point)
                pixel_features = pixel_features[:, 0, :]
                return pixel_features

            rgb_points = tf.cast(rgb_points, tf.float32)
            # TODO REMOVE THIS ...URGENT...
            rgb_features = tf.map_fn(get_rgb_feats, rgb_points)
            
            distances = tf.cast(distances[:, :, :3], tf.float32)
            
            rgb_features = tf.concat([rgb_features, distances], 2)
            rgb_features = tf.cast(rgb_features, tf.float32)
            rgb_features = tf.transpose(rgb_features, [1, 0, 2])
            rgb_features = tf.reshape(rgb_features, [-1, int(rgb_features.shape[1]) * int(rgb_features.shape[2])])
            
            dense_features = MLP(rgb_features)
            out = out.write(h, dense_features)
            return tf.add(h, 0), tf.add(i, 0), tf.add(i, 0), tf.add(j, 1), out

        def fn_height(h, i, original_i, out1):
            print('height')
            j = tf.constant(0)
            out = tf.TensorArray(size=self.dense_bev_width, dtype=tf.float32, element_shape=[None, 1], infer_shape=True)
            while_j_condition = lambda b, m, n, l, _: tf.less(l, self.dense_bev_width)
            _, _,_ , _, result_width = tf.while_loop(while_j_condition, fn_width, [h, i, i * tf.cast(self.lidar_scale_height, tf.int32), j, out],
                                                # shape_invariants = [h.get_shape(), i.get_shape(), i.get_shape(), j.get_shape(), tf.TensorShape([None, 1])],
                                                return_same_structure=False)
            out1 = out1.write(i, result_width.stack())
            return tf.add(h, 0), tf.add(i, 1), tf.add(i, 1), out1

        def fn_channel(h, out1):
            print('channel')
            i = tf.constant(0)
            out = tf.TensorArray(size=self.dense_bev_height, dtype=tf.float32, element_shape=[self.dense_bev_width, None, 1], infer_shape=True)
            while_i_condition = lambda b, m, n, _: tf.less(m, self.dense_bev_height)
            _, _,_, result_height = tf.while_loop(while_i_condition, fn_height, [h, i, i * tf.cast(self.lidar_scale_height, tf.int32), out],
                                                    return_same_structure=False)
            # return tf.add(h, 1), tf.concat(result_height, 1)
            out1 = out1.write(h, result_height.stack())
            return tf.add(h, 1), out1


        h = tf.constant(0)
        out2 = tf.TensorArray(size=32, dtype=tf.float32, element_shape=[self.dense_bev_height, self.dense_bev_width, None , 1], infer_shape=True)
        while_h_condition = lambda h, _: tf.less(h, self.bev_ch)
        _, result_channel = tf.while_loop(while_h_condition, fn_channel, 
                                            [h, out2])

        
        print(result_channel)
        # dense_bev = tf.concat(result_channel, 1)
        dense_bev = result_channel.stack()
                    
        """
        This is an assumption that we have the same number of channels that are summed.
        This is not the best for 3d object detection as the height is not well handled
        A quick solutionis to use a weighted sum (1d conv)
        """
        # dense_bev = tf.reshape(dense_bev, [-1, dense_bev_height, dense_bev_width, bev_ch, 1])
        # dense_bev = tf.reduce_sum(dense_bev, 3)
        print(dense_bev)
        dense_bev = tf.reshape(dense_bev, [-1, self.dense_bev_height, self.dense_bev_width, self.bev_ch])
        dense_bev = conv(dense_bev, channels=1, kernel=1, stride=1, scope='conv_'+self.scope)

        return dense_bev


    # def call(self, input):
    #     def MLP(input_feats):
    #         layer_1 = tf.nn.relu(tf.add(tf.matmul(input_feats, self.w1), self.b1))
    #         layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.w2), self.b2))
    #         out_layer = tf.add(tf.matmul(layer_2, self.w3), self.b3)
    #         return out_layer

    #     def fn_width(h, i, original_i, j, out):
    #         original_j = j * tf.cast(self.lidar_scale_width, tf.int32)
    #         k_points = self.kdtree[:, tf.cast(original_i, tf.int32), tf.cast(original_j, tf.int32), tf.cast(h, tf.int32), :, :]
    #         current_point = tf.constant(np.array([[1]]), dtype=tf.int32)
    #         current_point = tf.concat([current_point, [[tf.cast(original_i, tf.int32)]], [[tf.cast(original_j, tf.int32)]], [[tf.cast(h, tf.int32)]], [[tf.cast(1, tf.int32)]]], 0)
    #         current_point = tf.cast(tf.reshape(current_point[:4, :], [1, 4]), tf.float16)

    #         distances = tf.transpose(tf.map_fn(lambda k_points_batch: k_points_batch - current_point, tf.cast(k_points, tf.float16)), [1, 0, 2])
    #         rgb_points = tf.map_fn(lambda k_point_: project_point_from_lidar_to_image(k_point_, self.Tr_velo_to_cam, self.R0_rect, self.P3, self.shift_h, self.shift_w), 
    #                                     tf.cast(tf.transpose(k_points, [1, 0, 2]), tf.float32))
    #         rgb_points = tf.cast(rgb_points, tf.int32)
        
            
    #         def get_rgb_feats(rgb_point):
    #             pixel_features = tf.map_fn(lambda rgb_point_: input[:, tf.cast(rgb_point_[0] / self.rgb_scale_height, tf.int32), tf.cast(rgb_point_[1]/self.rgb_scale_width, tf.int32), :], rgb_point)
    #             pixel_features = pixel_features[:, 0, :]
    #             return pixel_features

    #         rgb_points = tf.cast(rgb_points, tf.float32)
    #         # TODO REMOVE THIS ...URGENT...
    #         rgb_features = tf.map_fn(get_rgb_feats, rgb_points)
            
    #         distances = tf.cast(distances[:, :, :3], tf.float32)
            
    #         rgb_features = tf.concat([rgb_features, distances], 2)
    #         rgb_features = tf.cast(rgb_features, tf.float32)
    #         rgb_features = tf.transpose(rgb_features, [1, 0, 2])
    #         rgb_features = tf.reshape(rgb_features, [-1, int(rgb_features.shape[1]) * int(rgb_features.shape[2])])
            
    #         dense_features = MLP(rgb_features)

    #         return tf.add(h, 0), tf.add(i, 0), tf.add(i, 0), tf.add(j, 1), dense_features

    #     def fn_height(h, i, original_i, out1):
    #         j = tf.constant(0)
    #         out = tf.constant([[1], [1]], dtype=tf.float32)
    #         while_j_condition = lambda b, m, n, l, _: tf.less(l, self.dense_bev_width)
    #         _, _,_ , _, result_width = tf.while_loop(while_j_condition, fn_width, [h, i, i * tf.cast(self.lidar_scale_height, tf.int32), j, out],
    #                                             shape_invariants = [h.get_shape(), i.get_shape(), i.get_shape(), j.get_shape(), tf.TensorShape([None,None])],
    #                                             return_same_structure=False)
    #         return tf.add(h, 0), tf.add(i, 1), tf.add(i, 0), result_width

    #     def fn_channel(h, out1):
    #         i = tf.constant(0)
    #         out = tf.constant([[1], [1]], dtype=tf.float32)
    #         while_i_condition = lambda b, m, n, _: tf.less(m, self.dense_bev_height)
    #         _, _,_, result_height = tf.while_loop(while_i_condition, fn_height, [h, i, i * tf.cast(self.lidar_scale_height, tf.int32), out],
    #                                                 shape_invariants = [i.get_shape(), i.get_shape(), i.get_shape(), tf.TensorShape(None)],
    #                                                 return_same_structure=False)
    #         # return tf.add(h, 1), tf.concat(result_height, 1)
    #         return tf.add(h, 1), result_height


    #     h = tf.constant(0)
    #     out2 = tf.constant([[1], [1]], dtype=tf.float32)
    #     while_h_condition = lambda h, _: tf.less(h, self.bev_ch)
    #     _, result_channel = tf.while_loop(while_h_condition, fn_channel, 
    #                                         [h, out2],
    #                                         shape_invariants = [h.get_shape(), tf.TensorShape(None)])

        
    #     # dense_bev = tf.concat(result_channel, 1)
    #     dense_bev = result_channel
                    
    #     """
    #     This is an assumption that we have the same number of channels that are summed.
    #     This is not the best for 3d object detection as the height is not well handled
    #     A quick solutionis to use a weighted sum (1d conv)
    #     """
    #     # dense_bev = tf.reshape(dense_bev, [-1, dense_bev_height, dense_bev_width, bev_ch, 1])
    #     # dense_bev = tf.reduce_sum(dense_bev, 3)
    #     print(dense_bev)
    #     dense_bev = tf.reshape(dense_bev, [-1, self.dense_bev_height, self.dense_bev_width, self.bev_ch])
    #     dense_bev = conv(dense_bev, channels=1, kernel=1, stride=1, scope='conv_'+self.scope)

    #     return dense_bev


        
    # def call(self, input):
    #     def MLP(input_feats):
    #         layer_1 = tf.nn.relu(tf.add(tf.matmul(input_feats, self.w1), self.b1))
    #         layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.w2), self.b2))
    #         out_layer = tf.add(tf.matmul(layer_2, self.w3), self.b3)
    #         return out_layer

    #     def fn_width(h, i, original_i, j):
    #         original_j = j * self.lidar_scale_width
    #         k_points = self.kdtree[:, tf.cast(original_i, tf.int32), tf.cast(original_j, tf.int32), tf.cast(h, tf.int32), :, :]
    #         current_point = tf.constant(np.array([[1]]), dtype=tf.int32)
    #         current_point = tf.concat([current_point, [[tf.cast(original_i, tf.int32)]], [[tf.cast(original_j, tf.int32)]], [[tf.cast(h, tf.int32)]], [[tf.cast(1, tf.int32)]]], 0)
    #         current_point = tf.cast(tf.reshape(current_point[:4, :], [1, 4]), tf.float16)

    #         distances = tf.transpose(tf.map_fn(lambda k_points_batch: k_points_batch - current_point, tf.cast(k_points, tf.float16)), [1, 0, 2])
    #         rgb_points = tf.map_fn(lambda k_point_: project_point_from_lidar_to_image(k_point_, self.Tr_velo_to_cam, self.R0_rect, self.P3, self.shift_h, self.shift_w), 
    #                                     tf.cast(tf.transpose(k_points, [1, 0, 2]), tf.float32))
    #         rgb_points = tf.cast(rgb_points, tf.int32)
        
            
    #         def get_rgb_feats(rgb_point):
    #             pixel_features = tf.map_fn(lambda rgb_point_: input[:, tf.cast(rgb_point_[0] / self.rgb_scale_height, tf.int32), tf.cast(rgb_point_[1]/self.rgb_scale_width, tf.int32), :], rgb_point)
    #             pixel_features = pixel_features[:, 0, :]
    #             return pixel_features

    #         rgb_points = tf.cast(rgb_points, tf.float32)
    #         # TODO REMOVE THIS ...URGENT...
    #         # rgb_points = tf.abs(rgb_points)
    #         rgb_features = tf.map_fn(get_rgb_feats, rgb_points)
            
    #         distances = tf.cast(distances[:, :, :3], tf.float32)
            
    #         rgb_features = tf.concat([rgb_features, distances], 2)
    #         rgb_features = tf.cast(rgb_features, tf.float32)
    #         rgb_features = tf.transpose(rgb_features, [1, 0, 2])
    #         rgb_features = tf.reshape(rgb_features, [-1, int(rgb_features.shape[1]) * int(rgb_features.shape[2])])
            
    #         dense_features = MLP(rgb_features)

    #         return dense_features

    #     def fn_height(h, i, original_i):
    #         result_width = tf.map_fn(lambda j: fn_width(h, i, i * self.lidar_scale_height, j), tf.constant(list(range(self.dense_bev_width)), dtype=tf.float32), dtype=tf.float32)
    #         return tf.concat(result_width, 1)

    #     def fn_channel(h):
    #         result_height = tf.map_fn(lambda i: fn_height(h, i, i * self.lidar_scale_height), tf.constant(list(range(self.dense_bev_height)), dtype=tf.float32), dtype=tf.float32)
    #         return tf.concat(result_height, 1)

    #     result_channel = tf.map_fn(lambda h: fn_channel(h), tf.constant(list(range(self.bev_ch)), dtype=tf.float32), dtype=tf.float32)
    #     dense_bev = tf.concat(result_channel, 1)
                    
    #     """
    #     This is an assumption that we have the same number of channels that are summed.
    #     This is not the best for 3d object detection as the height is not well handled
    #     A quick solutionis to use a weighted sum (1d conv)
    #     """
    #     # dense_bev = tf.reshape(dense_bev, [-1, dense_bev_height, dense_bev_width, bev_ch, 1])
    #     # dense_bev = tf.reduce_sum(dense_bev, 3)
        
    #     dense_bev = tf.reshape(dense_bev, [-1, self.dense_bev_height, self.dense_bev_width, self.bev_ch])
    #     dense_bev = conv(dense_bev, channels=1, kernel=1, stride=1, scope='conv_'+self.scope)

    #     return dense_bev