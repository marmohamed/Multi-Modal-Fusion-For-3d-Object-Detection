from ops.ops import *
from utils.utils import *

def FPN(layers_outputs, scope, is_training=True, reuse=False):

    with tf.variable_scope(scope, reuse=reuse):

        new_features = [128] * len(layers_outputs)
        new_layers = []
        prev_layer = None

        if 'rgb' in scope:
            channels = [64, 128, 192, 256]
        else:
            channels = [64] * 4

        for i in range(len(layers_outputs)-1, -1, -1):
            new_layer = conv(layers_outputs[i], new_features[i], kernel=1, stride=1, scope='conv0_' + str(i))
            new_layer = batch_norm(new_layer, is_training=is_training, scope='bn0_' + str(i))
            new_layer = relu(new_layer)
            new_layer = dropout(new_layer, rate=0.2, scope='new_layer_pre_fpn_' + str(i), training=is_training)

            if i != len(layers_outputs)-1:
                if new_layer.get_shape()[1] > prev_layer.get_shape()[1]:
                    prev_layer = upsample(prev_layer, scope='upsample_' + str(i), filters=64, use_deconv=True, kernel_size=4)

                new_layer_height = new_layer.get_shape()[1]
                prev_layer_height = prev_layer.get_shape()[1]
                diff = prev_layer_height - new_layer_height
                half_diff = diff//2

                new_layer_width = new_layer.get_shape()[2]
                prev_layer_width = prev_layer.get_shape()[2]
                diff_w = prev_layer_width - new_layer_width
                half_diff_w = diff_w//2

                new_layer = zeropad(new_layer, ((half_diff, diff-half_diff), (half_diff_w, diff_w-half_diff_w)), scope='zeropad_' + str(i))

                new_layer = tf.concat([new_layer, prev_layer], axis=-1)
                
                new_layer = conv(new_layer, new_features[i], kernel=3, stride=1, scope='conv1_' + str(i))
                new_layer = batch_norm(new_layer, is_training=is_training, scope='bn1_' + str(i))
                new_layer = relu(new_layer)
                new_layer = dropout(new_layer, rate=0.2, scope='new_layer_fpn_' + str(i), training=is_training)

            prev_layer = new_layer

            new_layer = conv(new_layer, channels[i], kernel=3, stride=1, scope='conv2_' + str(i))
            new_layer = batch_norm(new_layer, is_training=is_training, scope='bn2_' + str(i))
            new_layer = relu(new_layer)
            new_layer = dropout(new_layer, rate=0.2, scope='new_layer_post_fpn_' + str(i), training=is_training)

            new_layers.append(new_layer)

        return new_layers[::-1]
            
