from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import optimizer

GATE_OP = 1


class PCGrad(optimizer.Optimizer):
    '''Tensorflow implementation of PCGrad.
    Gradient Surgery for Multi-Task Learning: https://arxiv.org/pdf/2001.06782.pdf
    '''

    def __init__(self, optimizer, use_locking=False, name="PCGrad"):
        """optimizer: the optimizer being wrapped
        """
        super(PCGrad, self).__init__(use_locking, name)
        self.optimizer = optimizer

    # @tf.function
    def compute_gradients(self, loss, var_list=None,
                        gate_gradients=GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
        assert type(loss) is list
        num_tasks = len(loss)
        loss = tf.stack(loss)
        tf.random.shuffle(loss)
        loss_list = tf.unstack(loss)
        
        if var_list is None:
          var_list = (
          tf.trainable_variables() +
          ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        # else:
        #   var_list = tf.nest.flatten(var_list)
        #   var_list = tf.squeeze(var_list)


        grads_flag = {0: [], 1: [], 2: [], 3: []}
        temp = []
        for i in range(num_tasks):
          t1 = []
          x = loss_list[i]
          for grad in tf.gradients(x, var_list):
            if grad is not None:
              t1.append(tf.reshape(grad, [-1,]))
              grads_flag[i].append(True)
            else:
              grads_flag[i].append(False)
          temp.append(tf.concat(t1, axis=0))
        grads_task = tf.stack(temp)

        # Compute gradient projections.
        def proj_grad(grad_task):
            for k in range(num_tasks):
                inner_product = tf.reduce_sum(grad_task*grads_task[k])
                proj_direction = inner_product / tf.reduce_sum(grads_task[k]*grads_task[k])
                grad_task = grad_task - tf.minimum(proj_direction, 0.) * grads_task[k]
            return grad_task

        # proj_grads_flatten = tf.vectorized_map(proj_grad, grads_task)
        proj_grads_flatten = tf.map_fn(proj_grad, grads_task)

        # Unpack flattened projected gradients back to their original shapes.
        proj_grads = []
        for j in range(num_tasks):
            start_idx = 0
            k = 0
            for idx, var in enumerate(var_list):
              if grads_flag[j][k]:
                grad_shape = var.get_shape()
                flatten_dim = np.prod([grad_shape.dims[i].value for i in range(len(grad_shape.dims))])
                proj_grad = proj_grads_flatten[j][start_idx:start_idx+flatten_dim]
                proj_grad = tf.reshape(proj_grad, grad_shape)
                # if len(proj_grads) < len(var_list[grads_flag[j]]):
                proj_grads.append(proj_grad)
                # else:
                #     proj_grads[idx] += proj_grad               
                start_idx += flatten_dim
              k += 1
        grads_and_vars = list(zip(proj_grads, list(np.array(var_list)[grads_flag[0]])))
        return grads_and_vars

    def _create_slots(self, var_list):
        self.optimizer._create_slots(var_list)

    def _prepare(self):
        self.optimizer._prepare()

    def _apply_dense(self, grad, var):
        return self.optimizer._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, var):
        return self.optimizer._resource_apply_dense(grad, var)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        return self.optimizer._apply_sparse_shared(grad, var, indices, scatter_add)

    def _apply_sparse(self, grad, var):
        return self.optimizer._apply_sparse(grad, var)

    def _resource_scatter_add(self, x, i, v):
        return self.optimizer._resource_scatter_add(x, i, v)

    def _resource_apply_sparse(self, grad, var, indices):
        return self.optimizer._resource_apply_sparse(grad, var, indices)

    def _finish(self, update_ops, name_scope):
        return self.optimizer._finish(update_ops, name_scope)

    def _call_if_callable(self, param):
        """Call the function if param is callable."""
        return param() if callable(param) else param
