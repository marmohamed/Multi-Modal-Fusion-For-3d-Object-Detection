from utils.utils import *

from models.ResNet import *
from models.pytorch_to_tf import *

from ops.ops_pretrained import *

class Discriminator(object):

    def build_model(self, train_inptus, is_training=True, reuse=False, **kwargs):
        self.model = get_torch_model()
        self.__network(train_inptus, is_training, reuse)

    def __network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("network_discriminator", reuse=reuse):

            
            residual_list = get_residual_layer(18)

            ch = 64 
            x = conv2d(self.model.conv1, inp=x)

            
            x = batch_norm(self.model.bn1, inp=x)
            x = relu(inp=x)
            x = max_pool(self.model.maxpool, inp=x)

            x = resblock(x, [self.model.layer1[0].conv1, None, self.model.layer1[0].conv2], 
                                    [self.model.layer1[0].bn1, None, self.model.layer1[0].bn2],
                                    downsample=False, scope='resblock_10')                
            x = resblock(x, [self.model.layer1[1].conv1, None, self.model.layer1[1].conv2],
                                    [self.model.layer1[1].bn1, None, self.model.layer1[1].bn2],
                                    downsample=False, scope='resblock_11')


            x = resblock(x, [self.model.layer2[0].conv1, self.model.layer2[0].downsample[0],
                                self.model.layer2[0].conv2],
                                [self.model.layer2[0].bn1, self.model.layer2[0].downsample[1],
                                self.model.layer2[0].bn2], 
                                downsample=True, scope='resblock_21')
            x = resblock(x, [self.model.layer2[1].conv1, None, self.model.layer2[1].conv2],
                                [self.model.layer2[1].bn1, None, self.model.layer2[1].bn2], 
                                downsample=False, scope='resblock_22')


            x = resblock(x, 
                            [self.model.layer3[0].conv1, self.model.layer3[0].downsample[0], self.model.layer3[0].conv2], 
                            [self.model.layer3[0].bn1, self.model.layer3[0].downsample[1],
                                self.model.layer3[0].bn2],
                            downsample=True, scope='resblock_31')
            x = resblock(x, [self.model.layer3[1].conv1, None, self.model.layer3[1].conv2], 
                                    [self.model.layer3[1].bn1, None, self.model.layer3[1].bn2],
                            downsample=False, scope='resblock_32')


            x = resblock(x, 
                            [self.model.layer4[0].conv1, self.model.layer4[0].downsample[0], self.model.layer4[0].conv2], 
                            [self.model.layer4[0].bn1, self.model.layer4[0].downsample[1],
                                self.model.layer4[0].bn2],
                            downsample=True, scope='resblock_41')
            x = resblock(x, [self.model.layer4[1].conv1, None, self.model.layer4[1].conv2],
                                [self.model.layer4[1].bn1, None, self.model.layer4[1].bn2], 
                            downsample=False, scope='resblock_42')

            
            x = avg_pool(self.model.avgpool, inp=x)
            x = flatten(inp=x)
            # x = fc(self.model.fc, inp=x)
            x = fully_conneted_not_trained(x, 1)

            return x
