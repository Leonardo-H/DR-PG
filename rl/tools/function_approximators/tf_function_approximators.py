import tensorflow as tf
import numpy as np
from abc import abstractmethod
from rl.tools.function_approximators.function_approximator import FunctionApproximator, online_compatible

from rl.tools.normalizers import tfNormalizer, tfNormalizerMax
from rl.tools.utils import tf_utils as U
tfObject = U.tfObject
tf_float = U.tf_float


class tfFunctionApproximator(tfObject, FunctionApproximator):
    """
    An abstract helper class to define function approximators based on tensorflow.

    Note: The seed is for setting the operation-level seed. The randomness is
    tensorflow is determined jointly by graph-level seed and operation-level
    seed. If this operation-level seed is not set, the randomness can be
    different across devices.
    """

    @tfObject.save_init_args()
    def __init__(self, x_dim, y_dim, name='tfFunctionApproximator', seed=None,
                 build_nor=None, max_to_keep=None, bg_kwargs=None, pred_residue=False):

        if bg_kwargs is None:
            bg_kwargs = {}

        # whether to predict residue
        self._pred_residue = pred_residue
        if self._pred_residue:
            assert x_dim >= y_dim

        # just an abstract interface
        FunctionApproximator.__init__(self, x_dim, y_dim, name=name, seed=seed)
        # build normalizer (an tfObject)
        if build_nor is None:
            self._nor = tfNormalizerMax((self.x_dim,), unscale=False,
                                        unbias=False, clip_thre=5.0, rate=0., momentum=None)
        else:
            self._nor = build_nor((self.x_dim,))
            assert isinstance(self._nor, tfNormalizer)
        # build tf graph
        tfObject.__init__(self, name, max_to_keep=max_to_keep,
                          bg_kwargs=bg_kwargs)

    # Required methods of tfObject
    def _build_graph(self, **kwargs):
        """
        Builds the graph of mapping through the user-provided
        _build_func_apprx.  After all the tf.Variables are created it adds a
        new attribute _sh_vars (a Shaper object) for convenient manipulation of
        the tf.Variables inside the graph.

        Added attributes:
            ph_x, ts_nor_x, ts_y, _yh, _sh_vars
        """
        # build the input placeholder
        self.ph_x = tf.placeholder(shape=[None, self.x_dim], name="input", dtype=tf_float)
        # build the normalizer for whitening
        self.ts_nor_x = self._nor.build_nor_ops(self.ph_x)
        
        self.ts_yh = self._build_func_apprx(self.ts_nor_x, **kwargs)
        if self._pred_residue:
            self.ts_yh = self.ph_x[:, :self.y_dim] + self.ts_yh

        self._yh = U.function([self.ph_x], self.ts_yh)

        # build a Shaper of trainable variables for transforming
        # between continguous and list representations
        self._sh_vars = U.Shaper(self.ts_vars)

        # XXX
        # gradient wrt input
        ph_grad_ys = tf.placeholder(tf_float, shape=(None, self.y_dim), name='grad_ys')
        self._ts_grad_yh = tf.gradients(self.ts_yh, self.ph_x, grad_ys=ph_grad_ys)[0]
        self._grad_yh = U.function([self.ph_x, ph_grad_ys], self._ts_grad_yh)

        # Hessian-vector product wrt input
        ph_v = tf.placeholder(tf_float, shape=(None, self.x_dim), name='v')
        self._ts_hess_yh_v = tf.gradients(self._ts_grad_yh, self.ph_x, grad_ys=ph_v)[0]  # Hess*v
        self._hess_yh_v = U.function([self.ph_x, ph_v, ph_grad_ys], self._ts_hess_yh_v)

    @property
    def _pre_deepcopy_list(self):
        return tfObject._pre_deepcopy_list.fget(self) + ['_nor']

    # Required methods of FunctionApproximator
    # save, restore, copy, __deepcopy__ have been inherited from tfObject
    @online_compatible
    def predict(self, x):
        return self._yh(x)

    # XXX
    def grad(self, x, grad_ys=None):
        # x: batch * x_dim
        # grad_ys: y_dim, or batch * y_dim
        if grad_ys is None:
            grad_ys = np.ones((x.shape[0], self.y_dim))
        if len(grad_ys.shape)==1:
            grad_ys = np.tile(grad_ys,(x.shape[0],1))
        return self._grad_yh(x, grad_ys)  # batch * x_dim

    def grad_prod(self, x, M):
        # J^t * M
        # M: batch * y_dim * z_dim
        assert len(M.shape)==3
        z_dim = M.shape[2]
        JM = np.zeros((x.shape[0], x.shape[1], z_dim))
        for i in range(z_dim):
            JM[:,:,i] = self.grad(x, grad_ys=M[:,:,i])

        return JM  # batch * x_dim * z_dim


    def Hess_prod(self, x, v, grad_ys=None):
        #if len(v.shape)==1:
        #    v = np.tile(v,(x.shape[0],1))
        assert x.shape==v.shape
        # grad_ys: y_dim, or batch * y_dim
        if grad_ys is None:
            grad_ys = np.ones((x.shape[0], self.y_dim))
        if len(grad_ys.shape)==1:
            grad_ys = np.tile(grad_ys,(x.shape[0],1))
        return self._hess_yh_v(x, v, grad_ys)

    def Hess(self, x):
        H = np.zeros((x.shape[0], self.x_dim, self.x_dim))
        for i in range(self.x_dim):
            v = np.zeros(x.shape)
            v[:,i] = 1.0
            H[:,:,i] = self.Hess_prod(x, v)

        #Ht = np.transpose(H,(0,2,1))
        #assert np.isclose(np.sum(H-Ht), 0.0)
        return H


    def prepare_for_update(self, x):
        self._nor.update(x)

    @property
    def variable(self):
        return self._sh_vars.variable

    @variable.setter
    def variable(self, val):
        self._sh_vars.variable = val

    def assign(self, other):
        # need to overload this because of self._nor
        assert type(self) == type(other)
        tfObject.assign(self, other)
        self._nor.assign(other._nor)

    # Methods to be implemented
    @abstractmethod
    def _build_func_apprx(self, ts_nor_x, **kwargs):
        """ Return the prediction given normalized input ts_nor_x as a
        tf.Tensor. The extra arguments args and kwargs can be passed as
        bg_kwargs in __init__. """


class tfMLPFunctionApproximator(tfFunctionApproximator):

    @tfObject.save_init_args()
    def __init__(self, x_dim, y_dim, n_layers, size, name='tfMLPFunctionApproximator',
                 seed=None, build_nor=None, max_to_keep=None, pred_residue=False, activation=tf.tanh):
        # setting for MLP
        self._n_layers = n_layers
        self._size = size
        self._activation = activation
        super().__init__(x_dim, y_dim, name=name, seed=seed,
                         build_nor=build_nor, max_to_keep=max_to_keep, pred_residue=pred_residue)

    def _build_func_apprx(self, ts_nor_x):
        return U.build_multilayer_perceptron('yhat', ts_nor_x, self.y_dim, self._n_layers, self._size, self._activation, output_init_std=0.01)

    def _rebuild_func_apprx_with_raw(self, x):
        ts_nor_x = self._nor.build_nor_ops(x)
        with tf.variable_scope(self._tfObject__name):
            assert tf.get_variable_scope().name == self._tfObject__scope
            return U.build_multilayer_perceptron('yhat', ts_nor_x, self.y_dim, self._n_layers, self._size, self._activation, output_init_std=0.01, reuse=True)

    def _rebuild_func_apprx(self, x):
        ts_nor_x = x
        with tf.variable_scope(self._tfObject__name):
            assert tf.get_variable_scope().name == self._tfObject__scope
            return U.build_multilayer_perceptron('yhat', ts_nor_x, self.y_dim, self._n_layers, self._size, self._activation, output_init_std=0.01, reuse=True)
