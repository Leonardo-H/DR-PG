import collections
import tensorflow as tf
import numpy as np
from functools import wraps
from abc import abstractmethod

from rl.policies.policy import Policy
from rl.tools.function_approximators import tfFunctionApproximator, tfMLPFunctionApproximator, online_compatible
from rl.tools.utils.misc_utils import zipsame
from rl.tools.utils import tf_utils as U
from rl.tools.utils.misc_utils import flatten
tfObject = U.tfObject
tf_float = U.tf_float


class tfPolicy(tfFunctionApproximator, Policy):
    """
    An abstract helper class to define policies based on tensorflow.

    Note: The seed is for setting the operation-level seed. The randomness is
    tensorflow is determined jointly by graph-level seed and operation-level
    seed. If this operation-level seed is not set, the randomness can be
    different across devices.
    """

    @tfObject.save_init_args()
    def __init__(self, x_dim, y_dim, name='tfPolicy', seed=None,
                 build_nor=None, max_to_keep=None, bg_kwargs=None):

        tfFunctionApproximator.__init__(self, x_dim, y_dim, name=name, seed=seed,
                                        build_nor=build_nor, max_to_keep=max_to_keep,
                                        bg_kwargs=bg_kwargs)
        # so we don't build repetitive operators
        self._kl_cache = collections.defaultdict(lambda: None)

    @property
    def ph_obs(self):
        return self.ph_x

    @property
    def ph_acs(self):
        return self.ph_y

    # required methods of tfFunctionApproximator
    def _build_graph(self, **kwargs):
        """ We treat tfFunctionApproximator as the stochastic map of the policy
        (which inputs ph_x and outputs ts_yh) and build additional
        attributes/methods required by Policy """
        # build tf.Variables
        # add attributes ph_x, ts_nor_x, ts_y, _yh, _sh_vars,
        #                ph_y, ts_pi, ts_logp, ts_pid, ts_pir, ts_pi_given_r
        tfFunctionApproximator._build_graph(self, **kwargs)
        # r_dim: dimension of randomness in generating actions.
        # build additional graphs for Policy
        # build conditional distribution
        self._pi = self._yh
        self._pi_given_r = U.function([self.ph_x, self.ph_r], self.ts_pi_given_r)
        self._pid = U.function([self.ph_x], self.ts_pid)  # derandomized actions
        # actions and the randomness used in generating actions concatenated.
        self._pir = U.function([self.ph_x], self.ts_pir)
        self._logp = U.function([self.ph_x, self.ph_y], self.ts_logp)
        self._logp_grad = U.function([self.ph_x, self.ph_y], tf.gradients(self.ts_logp, self.ts_vars))
        # build fvp operator (this depends only on self)
        ph_g, ts_grads = self._sh_vars.build_flat_ph()
        ts_kl = self.build_kl(self, self, p1_sg=True)
        ts_kl_grads = U.gradients(ts_kl, self.ts_vars)  # grad to the 2nd arg of KL
        ts_inner_prod = tf.add_n([tf.reduce_sum(kg * v) for (kg, v) in zipsame(ts_kl_grads, ts_grads)])
        ts_fvp = U.gradients(ts_inner_prod, self.ts_vars)  # Fisher (information matrix) and Vector Product
        ts_fvp = tf.concat([tf.reshape(f, [-1]) for f in ts_fvp], axis=-1)  # continuous vector
        self._fvp = U.function([self.ph_x, ph_g], ts_fvp)

        # build nabla logp f.
        ts_loss = tf.reduce_sum(self.ph_f * self.ts_logp)  # sum!!
        ts_grads = U.gradients(ts_loss, self.ts_vars)
        ts_grads = [g if g is not None else tf.zeros_like(v) for (v, g) in
                    zipsame(self.ts_vars, ts_grads)]
        # need to flatten
        compute_ts_grad = U.function([self.ph_x, self.ph_y, self.ph_f], ts_grads)
        self.nabla_logp_f = lambda x, y, f: flatten(compute_ts_grad(x, y, f))

    def _build_func_apprx(self, ts_nor_x, **kwargs):
        # required method called inside tfFunctionApproximator._build_graph
        # add attributes ph_y, ts_pi, ts_logp, ts_pid
        self.ph_r = tf.placeholder(shape=[None, self.r_dim], name='randomness', dtype=tf_float)
        self.ph_y = tf.placeholder(shape=[None, self.y_dim], name="output", dtype=tf_float)
        self.ph_f = tf.placeholder(shape=[None], name='function', dtype=tf_float)
        # user-defined
        self.ts_pi, self.ts_logp, self.ts_pid, self.ts_pir, self.ts_pi_given_r = self._build_dist(
            ts_nor_x, self.ph_y, **kwargs)
        if self.ts_pid is None:
            self.ts_pid = self.ts_pi
        return self.ts_pi

    # required methods of Policy
    @online_compatible
    def pi(self, x, stochastic=True):
        return self._pi(x) if stochastic else self._pid(x)

    @online_compatible
    def pid(self, x):
        return self._pid(x)

    @online_compatible
    def pir(self, x):
        return self._pir(x)

    @online_compatible
    def pi_given_r(self, x, r):
        return self._pi_given_r(x, r)

    @online_compatible
    def logp(self, x, y):
        return self._logp(x, y)

    def logp_grad(self, x, y):
        # XXX Only support single instance, due to tf can not compute jacobian.
        # Should throw error if more than one instances are provided.
        # Return a flat np array.
        return flatten(self._logp_grad(x[None], y[None]))

    def kl(self, other, x, reversesd=False):
        assert type(other) == type(self)
        key = str(id(other)) + str(reversesd)
        if self._kl_cache[key] is None:
            ts_kl = self.build_kl(self, other) if reversesd else self.build_kl(other, self)
            _kl = U.function([self.ph_x, other.ph_x], ts_kl)
            self._kl_cache[key] = lambda _x: _kl(_x, _x)
        return self._kl_cache[key](x)

    def fvp(self, x, g):
        return self._fvp(x, g)

    # methods to be implemented
    @abstractmethod
    def _build_dist(self, ts_x, ph_y, **kwargs):
        """ Return pi, logp, pid, and pir tf.Tensors, which represents the sampled
        action, the derandomized action, and the log probability of a sampled
        action-state pair. If the policy is fully stochastic, pid can be set as
        None.
        """

    @classmethod
    @abstractmethod
    def build_kl(cls, p1, p2, p1_sg=False, p2_sg=False):
        """return KL(p1||p2) as a tf.Tensor, where p1 and p2 are objects of
        this class, and p1_sg and p2_sg denotes whether to treat p1 and p2 as
        constants, respectively, so the gradient is stopped properly when
        compute_gradient is called """


class tfGaussianPolicy(tfPolicy):
    """
    An abstract class which is namely tfPolicy with Gaussian distribution based
    on tfFunctionApproximator.
    """
    @staticmethod
    def _build_logp(dim, rvs, mean, logstd):  # log probability of Gaussian
        return (-0.5 * U.squared_sum((rvs - mean) / tf.exp(logstd), axis=1) -
                tf.reduce_sum(logstd) - 0.5 * np.log(2.0 * np.pi) * tf.to_float(dim))

    @classmethod
    def build_kl(cls, p1, p2, p1_sg=False, p2_sg=False, w=None):
        """KL(p1, p2). p1_sg: whether stop gradient for p1."""
        def get_attr(p, sg):
            logstd, mean = p.ts_logstd, p.ts_mean
            if sg:
                logstd, mean = tf.stop_gradient(logstd), tf.stop_gradient(mean)
            std = tf.exp(logstd)
            return logstd, std, mean
        logstd1, std1, mean1 = get_attr(p1, p1_sg)
        logstd2, std2, mean2 = get_attr(p2, p2_sg)
        kl = logstd2 - logstd1 - 0.5 * p1.y_dim   # XXX fixed the bug
        kl += (tf.square(std1) + tf.square(mean1 - mean2)) / (2.0 * tf.square(std2))
        kl = tf.reduce_sum(kl, axis=-1)  # reduce over ac dimension
        if w is None:
            kl = tf.reduce_mean(kl)  # average over x
        else:  # weighted average over x
            kl = tf.reduce_sum(kl * w) / tf.reduce_sum(w)
        return kl


def _tfGaussianPolicyDecorator(cls):
    """
    A decorator for defining tfGaussianPolicy via tfFunctionApproximator.
    It is mainly tfGaussianPolicy but uses cls' _build_func_apprx.
    """
    assert issubclass(cls, tfFunctionApproximator)

    class decorated_cls(tfGaussianPolicy, cls):

        @tfObject.save_init_args()
        def __init__(self, x_dim, y_dim, init_std,
                     name='tfGaussianPolicy', seed=None,
                     build_nor=None, max_to_keep=None,
                     min_std=0.1, **kwargs):
            # Since tfPolicy only modifies the __init__ of tfFunctionApproximator
            # in adding _kl_cache, we reuse the __init__ of cls.

            # new attributes
            self.r_dim = y_dim  # dimension of randomness
            self.ts_mean = self.ts_logstd = self._ts_stop_std_grad = None
            self._init_std = init_std
            self._min_std = min_std

            # construct a tfFunctionApproximator
            cls.__init__(self, x_dim, y_dim, name=name, seed=seed,
                         build_nor=build_nor, max_to_keep=max_to_keep,
                         **kwargs)

            # add the cache of tfPolicy
            self._kl_cache = collections.defaultdict(lambda: None)

        @property
        def std(self):  # for convenience
            return self._std()

        @std.setter
        def std(self, val):
            self._set_logstd(np.log(val))

        def _rebuild_cls_func_apprx_with_raw(self, x, add=None):
            ts_nor_x = self._nor.build_nor_ops(x)
            return cls._rebuild_func_apprx(self, ts_nor_x)

        def stop_std_grad(self, cond=True):
            self._set_stop_std_grad(cond)

        def _build_dist(self, ts_nor_x, ph_y):
            # mean and std
            self.ts_mean = cls._build_func_apprx(self, ts_nor_x)  # use the tfFunctionApproximator to define mean
            self._ts_logstd = tf.get_variable(
                'logstd', shape=[self.y_dim], initializer=tf.constant_initializer(
                    np.log(self._init_std)))
            self._ts_stop_std_grad = tf.get_variable('stop_std_grad', initializer=tf.constant(False), trainable=False)
            _ts_logstd = tf.cond(self._ts_stop_std_grad,  # whether to stop gradient
                                 true_fn=lambda: tf.stop_gradient(self._ts_logstd),
                                 false_fn=lambda: self._ts_logstd)
            # make sure the distribution does not degenerate
            self.ts_logstd = tf.maximum(tf.to_float(np.log(self._min_std)), _ts_logstd)
            ts_std = tf.exp(self.ts_logstd)
            self._std = U.function([], ts_std)
            self._set_logstd = U.build_set([self._ts_logstd])
            self._set_stop_std_grad = U.build_set([self._ts_stop_std_grad])

            # pi
            # self.ts_noise = tf.random_normal(tf.shape(ts_std), stddev=ts_std, seed=self.seed)
            rand = tf.random_normal(tf.shape(self.ts_mean), seed=self.seed)
            ts_noise = ts_std * rand
            ts_pi = self.ts_mean + ts_noise
            ts_pid = self.ts_mean
            # Need to broadcast noise to each row.
            # n = tf.shape(self.ts_mean)[0]
            # noise = tf.reshape(tf.tile(self.ts_noise, [n]), [n, -1])
            # ts_pir = tf.concat([ts_pi, noise], 1)
            ts_pir = tf.concat([ts_pi, rand], axis=1)
            ts_pi_given_r = self.ts_mean + ts_std * self.ph_r
            # logp
            ts_logp = self._build_logp(self.y_dim, ph_y, self.ts_mean, self.ts_logstd)

            # XXX Full matrix
            # expectation with a quadratic function 0.5 y^t A y + b^t y + c
            ph_A = tf.placeholder(tf_float, shape=(None, self.y_dim, self.y_dim), name='A')
            ph_b = tf.placeholder(tf_float, shape=(None, self.y_dim, 1), name='b')
            ph_c = tf.placeholder(tf_float, shape=(None,), name='c')
            ph_w = tf.placeholder(tf_float, shape=(None,), name='w')

            ts_mean = tf.expand_dims(self.ts_mean, -1)  # None * dim_y * 1
            ts_var =  tf.reshape(ts_std**2, (1,self.y_dim,1)) # None * dim_y * 1

            ts_quad_exp = tf.squeeze(0.5*tf.matmul(ts_mean, tf.matmul(ph_A, ts_mean), adjoint_a=True)) \
                         + tf.squeeze(tf.matmul(ph_b, ts_mean,adjoint_a=True)) \
                         + ph_c + 0.5*tf.trace(ph_A*ts_var)  # None
            self._quad_exp = U.function([self.ph_x, ph_A, ph_b, ph_c], ts_quad_exp)
            ts_quad_exp_grads = U.gradients(ts_quad_exp*ph_w, self.ts_vars)  # just a vector
            compute_ts_quad_exp_grad = U.function([self.ph_x, ph_A, ph_b, ph_c, ph_w], ts_quad_exp_grads)
            self._nabla_quad_exp = lambda x, A, b, c, w: flatten(compute_ts_quad_exp_grad(x, A, b, c, w))

            # XXX Diagonal
            ph_A_diag = tf.placeholder(tf_float, shape=(None, self.y_dim, 1), name='A_diag')
            ts_quad_exp_diag = tf.squeeze(0.5*tf.matmul(ts_mean, ph_A_diag*ts_mean, adjoint_a=True)) \
                               + tf.squeeze(tf.matmul(ph_b, ts_mean,adjoint_a=True)) \
                               + ph_c + 0.5*tf.squeeze(tf.reduce_sum(ph_A_diag*ts_var, axis=1))  # None
            self._quad_exp_diag = U.function([self.ph_x, ph_A_diag, ph_b, ph_c], ts_quad_exp_diag)
            ts_quad_exp_grads_diag = U.gradients(ts_quad_exp_diag*ph_w, self.ts_vars)  # just a vector
            compute_ts_quad_exp_grad_diag = U.function([self.ph_x, ph_A_diag, ph_b, ph_c, ph_w], ts_quad_exp_grads_diag)
            self._nabla_quad_exp_diag = lambda x, A, b, c, w: flatten(compute_ts_quad_exp_grad_diag(x, A, b, c, w))


            return ts_pi, ts_logp, ts_pid, ts_pir, ts_pi_given_r

        # XXX
        def quad_exp(self, x, A, b, c, w=None):
            # expectation with a quadratic function 0.5 y^t A y + b^t y + c
            # A: batch * dim_y * dim_y or batch * dim_y
            # b: batch * dim_y or batch * dim_y * 1
            # c: batch
            # w: batch

            assert len(A.shape) in [2,3]
            diag = len(A.shape)==2
            if diag: 
                A = A[:,:,None]


            if len(b.shape)==2:
                b = np.expand_dims(b,-1)
            assert len(b.shape)==3
            assert len(c.shape)==1
            assert x.shape[0]==A.shape[0]==b.shape[0]==c.shape[0]
            # batch
            val = self._quad_exp_diag(x, A, b, c) if diag \
                    else self._quad_exp(x, A, b, c)  
            grad = None
            if w is not None:
                # compute gradient too
                assert len(w)==len(c)
                grad = self._nabla_quad_exp_diag(x, A, b, c, w) if diag \
                        else self._nabla_quad_exp(x, A, b, c, w)
            return val, grad


    # to make them look the same as intended
    decorated_cls.__name__ = cls.__name__
    decorated_cls.__qualname__ = cls.__qualname__
    return decorated_cls


@_tfGaussianPolicyDecorator
class tfGaussianMLPPolicy(tfMLPFunctionApproximator):
    pass
