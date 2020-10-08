import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np

# https://github.com/leonidk/pytorch-tf/blob/master/pytorch-tf.ipynb

def ws_reg(kernel):
    kernel_mean = tf.math.reduce_mean(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_mean')
    kernel = kernel - kernel_mean
    # kernel_std = tf.math.reduce_std(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_std')
    kernel_std = tf.keras.backend.std(kernel, axis=[0, 1, 2], keepdims=True)
    kernel = kernel / (kernel_std + 1e-5)


def conv2d(c,**kwargs):
    padding = 'VALID' if c.padding[0] is 0 else 'SAME'

    x = kwargs['inp']
    if c.padding[0] > 0:
        x = tf.keras.layers.ZeroPadding2D(padding=(c.padding[0], c.padding[1]))(x)

    padding = 'valid'
    
    W = c.weight.data.numpy().transpose([2, 3, 1, 0])
    if c.bias:
        b = c.bias.data.numpy()

    if c.bias:
        x = tf.keras.layers.Conv2D(c.out_channels, c.kernel_size, 
                                    strides=[c.stride[0], c.stride[1]],
                                    padding=padding,
                                    kernel_regularizer=ws_reg,
                                    kernel_initializer=tf.constant_initializer(W),
                                    bias_initializer=tf.constant_initializer(b), 
                                    use_bias=c.bias)(x)
    else:
        x = tf.keras.layers.Conv2D(c.out_channels, c.kernel_size, 
                                    strides=[c.stride[0], c.stride[1]],
                                    padding=padding,
                                    kernel_regularizer=ws_reg,
                                    kernel_initializer=tf.constant_initializer(W), 
                                    use_bias=c.bias)(x)

    return x

def relu(**kwargs):
    return tf.nn.leaky_relu(kwargs['inp'])
    
def max_pool(c,**kwargs):
    padding = 'VALID' if c.padding is 0 else 'SAME'
    x = tf.nn.max_pool(kwargs['inp'],[1,c.kernel_size,c.kernel_size,1],strides=[1,c.stride,c.stride,1],padding=padding)
    return x

def avg_pool(c,**kwargs):
    padding = 'VALID' if c.padding is 0 else 'SAME'
    x = tf.nn.avg_pool(kwargs['inp'],[1,c.kernel_size,c.kernel_size,1],strides=[1,c.stride,c.stride,1],padding=padding)
    return x

def flatten(**kwargs):
    x = tf.keras.layers.GlobalAveragePooling2D()(kwargs['inp'])
    return x

def fc(c, **kwargs):
    W = c.weight.data.numpy().transpose([1, 0])
    # if c.bias:
    b = c.bias.data.numpy()
    # if c.bias:
    x = tf.keras.layers.Dense(c.out_features, 
                                kernel_initializer=tf.constant_initializer(W),
                                bias_initializer=tf.constant_initializer(b),
                                use_bias=True)(kwargs['inp'])
    # else:
    #     x = tf.keras.layers.Dense(c.out_features, 
    #                             kernel_initializer=tf.constant_initializer(W),
    #                             use_bias=c.bias)(kwargs['inp'])
    return x

def fully_conneted_not_trained(x, units, use_bias=True, scope='fully_0'):
    with tf.variable_scope(scope):
        weight_init = tf.contrib.layers.xavier_initializer()
        weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)

        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x

def dropout(c,**kwargs):
    return kwargs['inp']

def batch_norm(c, is_training=True, **kwargs):
    # parameters = [p for p in c.parameters()]
    # beta = tf.constant_initializer(c.bias.data.numpy())
    # gamma = tf.constant_initializer(c.weight.data.numpy())
    # running_mean = tf.constant_initializer(c.running_mean.data.numpy())
    # running_var = tf.constant_initializer(c.running_var.data.numpy())
    # x = tf.layers.batch_normalization(kwargs['inp'], epsilon=c.eps, momentum=c.momentum,
    #                                 beta_initializer=beta,
    #                                 gamma_initializer=gamma,
    #                                 moving_mean_initializer = running_mean,
    #                                 moving_variance_initializer = running_var,
    #                                 training=is_training
    #                                 )
    # return x
    return group_norm(kwargs['inp'], G=32, eps=1e-5, scope=kwargs['scope'])

def group_norm(x, G=32, eps=1e-5, scope='group_norm') :
    with tf.variable_scope(scope) :
        N, H, W, C = x.get_shape().as_list()
        # print(N, H, W, C)
        G = min(G, C)

        x = tf.reshape(x, [-1, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)

        gamma = tf.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))


        x = tf.reshape(x, [-1, H, W, C]) * gamma + beta

    return x

def resblock(x_init, parameters, bns, is_training=True, scope='resblock', downsample=False) :
    with tf.variable_scope(scope) :

        x = conv2d(parameters[0], inp=x_init)
        x = batch_norm(bns[0], is_training=is_training, inp=x, scope=scope+'_0')
        x = relu(inp=x)
    
        x = conv2d(parameters[2], inp=x)
        x = batch_norm(bns[2], is_training=is_training, inp=x, scope=scope+'_1')

        if downsample :
            x_init = conv2d(parameters[1], inp=x_init)
            x_init = batch_norm(bns[1], is_training=is_training, inp=x_init, scope=scope+'_2')

        x = x + x_init
        x = relu(inp = x)

        return x

def bottle_resblock(x_init, parameters) :
    with tf.variable_scope(scope) :
        x = batch_norm(inp=x_init)
        shortcut = relu(inp=x)

        x = conv2d(parameters[0], inp=shortcut)
        x = batch_norm(inp=x)
        x = relu(inp=x)

        if downsample :
            x = conv2d(parameters[1], inp=x)
            shortcut = conv2d(parameters[2], inp=shortcut)

        else :
            x = conv2d(parameters[1], inp=x)
            shortcut = conv2d(parameters[2], inp=shortcut)

        x = batch_norm(inp=x)
        x = relu(inp=x)
        x = conv2d(parameters[3], inp=x)

        return x + shortcut



def get_residual_layer(res_n) :
    return [2, 2, 2, 2]



