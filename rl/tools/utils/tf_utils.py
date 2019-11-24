import collections
import copy
import tensorflow as tf
import numpy as np
import os
import multiprocessing
import functools
from abc import ABC, abstractmethod

from rl.tools.utils.misc_utils import unflatten, flatten, cprint

tf_float = tf.float32
tf_int = tf.int32


"""
For compatibility with stop_gradient
"""

def tf_flatten(variables):
    return tf.concat([tf.reshape(var, [-1]) for var in variables], axis=0)


def gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var)
            for var, grad in zip(var_list, grads)]


"""
Wrapper of tensorflow graphs
"""


class tfObjectManager(object):
    """
    An object manager that makes sure each one has an unique name.
    """

    def __init__(self, default_name='tfObject'):
        self._name = default_name
        self._dict = collections.defaultdict(lambda: None)
        self._table = collections.defaultdict(lambda: False)

    def get_name(self, name=None):
        """ automatically get a unique name for the constructing a tfObject instance """
        if name is None:
            name = self._name
        name = str(name)
        valid_name = False
        while not valid_name:
            # propose a new name
            ind = self._dict[name]
            if ind is None:
                new_name = str(name)
                self._dict[name] = 1
            else:
                new_name = str(name) + '_' + str(ind)
                self._dict[name] += 1
            # check if the proposed name exists
            if not self._table[new_name]:
                self._table[new_name] = True
                valid_name = True
        if name != new_name:
            cprint('An tfObject under {} already exists. A new name {} is created by tfObjectManager'.format(name, new_name))
        return new_name


# This makes sure every tfObject instance has an unique name
_tfOBJECT_MANAGER = tfObjectManager()


class tfObject(ABC):
    """
    A helper class for defining custom classes based on tensorflow.

    It makes sure that each instance of tfObject has an unique name (realized
    as tf.variable_scope) and support basic functionalities, like
    copy.deepcopy, assign, save, and restore.

    Usage guideilnes:

        The user needs to define _build_graph and use tfObject.save_init_args
        to decorate its __init__. Note stateful non-tensorflow attributes
        (which change during the use of the instance, like a counter) should be
        NOT created inside _build_graph. The decorator tfObject.save_init_args
        is used to record input arguments to __init__ for deepcopying. It
        should be used to decorate a child class's __init__ when its signature
        or default value changes.

        In order to maintain desired deepcopy behaviors during inheritance, the
        vuser should modify _pre_deepcopy_list and _post_deepcopy_list methods
        to to add the name of attributes that should be copied during deepcopy.
        By default, an tfObject instance does not deepcopy any attribute,
        except for those provided by the user. This convention is chosen for
        robust behvaiors in case of potential furture behavior changes of
        tensorflow. When copy.deepcopy is called, tfObject calls the __init__
        function defined by the custom class, in which before _build_graph is
        called (through __init__ of tfObject) the attributes in
        _pre_deepcopy_list will be deepcopied, and then deepcopies the
        attributes in the _post_deepcopy_list. As a rule of thumb,
        _pre_deepcopy_list should contain stateful attributes that pertain to
        the tensorflow graph creation (i.e. those created before calling
        __init__ )  _post_deepcopy_list contains other stateful attributes.

        Note when defining _pre_deepcopy and _post_deepcopy_list, make sure it
        contains the contents from the parent class.

    Public attributes:
        ts_vars, ts_allvars

    Public methods:
        copy, __deepcopy__, assign, save, restore
    """

    def __init__(self, name='tfObject', max_to_keep=None, bg_kwargs=None):
        """
            The tensorflow graph constructor.

            Args:
                name: the name of the object
                max_to_keep: the maximal number of copies to save
                bg_kwargs: the additional kwargs of _build_graph.
        """

        assert hasattr(self, '_tfObject__args') and hasattr(self, '_tfObject__kwargs'), \
            'Must use save_init_args decorator on __init__'
        if bg_kwargs is None:
            bg_kwargs = {}

        # pre-processing
        if hasattr(self, '_tfObject__pre_init_fun'):
            self._tfObject__pre_init_fun()
        if hasattr(self, '_tfObject__default_name'):
            name = self._tfObject__default_name  # force using a default name

        # create the tensorflow graph
        self.__name = _tfOBJECT_MANAGER.get_name(name)  # get a unique name
        with tf.variable_scope(self._tfObject__name):
            self.__scope = tf.get_variable_scope().name  # for later tensors retrieval
            self._build_graph(**bg_kwargs)
            # build getters and setters (np.ndarray)
            if len(self.ts_vars) > 0:
                self.__get_vars = build_get(self.ts_vars)
                self.__set_vars = build_set(self.ts_vars)
            if len(self.ts_allvars) > 0:
                self.__get_allvars = build_get(self.ts_allvars)
                self.__set_allvars = build_set(self.ts_allvars)
        if len(self.ts_allvars) > 0:
            self.__saver = tf.train.Saver(self.ts_allvars, max_to_keep=max_to_keep)

        # for deepcopy
        self.__pre_deepcopy_list = []  # attributes should be deep copied
        self.__pre_deepcopy_list.extend(self._pre_deepcopy_list)
        self.__post_deepcopy_list = ['_scope']  # attributes should be deep copied
        self.__post_deepcopy_list.extend(self._post_deepcopy_list)

    # Some functions for the user to define
    @abstractmethod
    def _build_graph(self, **kwargs):
        """ Build the tensorflow graph """

    @property
    def _pre_deepcopy_list(self):
        """ Return a list of strings of attribute names that should be deep
        copied before calling tfObject.__init__ """
        return []

    @property
    def _post_deepcopy_list(self):
        """ Return a list of strings of attribute names that should be deep
        copied before calling self.__init__ """
        return []

    # Functions for correct deepcopy
    @staticmethod
    def save_init_args(deepcopy_args=False):
        """ A decorator for child class's __init__, which saves the input
        arguments for performing deepcopying"""
        if deepcopy_args:  # whether to deepcopy the input arguments
            def safe_copy(val):
                try:
                    return copy.deepcopy(val)
                except:
                    return copy.copy(val)
        else:
            def safe_copy(val): return val

        def decorator(fun):
            @functools.wraps(fun)
            def wrapper(self, *args, **kwargs):
                if hasattr(self, '_tfObject__args_saved'):
                    if self._tfObject__args_saved:  # make sure it's only called once
                        return fun(self, *args, **kwargs)

                # save the input arguments
                self.__args, self.__kwargs = [], {}
                self.__args = [safe_copy(arg) for arg in args]
                self.__kwargs = {k: safe_copy(v) for k, v in kwargs.items()}
                self.__args_saved = True

                return fun(self, *args, **kwargs)
            return wrapper
        return decorator

    def copy(self, new_name):
        """ Like copy.deepcopy but with a new custom name """
        return self.__deepcopy(name=new_name, memo={})

    def __deepcopy__(self, memo):
        # we need to overload this because of tensorflow graph
        return self._tfObject__deepcopy(memo=memo)

    def __deepcopy(self, memo, name=None):
        # create new instance
        tfobj = type(self).__new__(type(self), *self._tfObject__args, **self._tfObject__kwargs)

        memo[id(self)] = tfobj  # prevent forming a loop
        # customize the behavior of tfObject.__init__
        if name is not None:
            tfobj.__default_name = name  # use a new name

        def _pre_deepcopy():  # deepcopy attributes before _build_graph
            tfobj._tfObject__update_attrs(self, self._tfObject__pre_deepcopy_list, memo)
        tfobj.__pre_init_fun = _pre_deepcopy
        # initialize the instance as usual
        tfobj.__init__(*self._tfObject__args, **self._tfObject__kwargs)
        # post deepcopying
        tfobj._tfObject__update_attrs(self, self._tfObject__post_deepcopy_list, memo)
        # update tf.Variables
        tfobj.assign(self)
        return tfobj

    def __update_attrs(self, src, attrs, memo):
        # try to deepcopy attrs from src to self
        for k in list(set(attrs) & set(src.__dict__.keys())):
            setattr(self, k, copy.deepcopy(getattr(src, k), memo))

    # Miscellaneous functions
    def assign(self, other):
        """Set the tf.Variables of self as that of the other """
        assert type(self) == type(other)
        if len(self.ts_allvars) > 0:
            self._tfObject__set_allvars(*other._tfObject__get_allvars())  # update tf.Variables

    @property
    def ts_vars(self):  # list of trainable tf.Variables
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._tfObject__scope)

    @ts_vars.setter
    def ts_vars(self, vals):  # list of values to set to trainable tf.Variables
        self._tfObject__set_vars(*vals)

    @property
    def ts_allvars(self):  # list of all tf.Variables, including non-trainable ones
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self._tfObject__scope)

    @ts_allvars.setter
    def ts_allvars(self, vals):  # list of all tf.Variables, including non-trainable ones
        self._tfObject__set_allvars(*vals)

    # TODO make this object pickle-compatible?
    def save(self, path):
        """ Save the ts_allvars to path """
        if len(self.ts_allvars) > 0:
            path = self._tfObject__saver.save(tf.get_default_session(), path)
        return path

    def restore(self, path, saved_name=None):
        """Recover ts_allvars from path saved with saved_name"""
        if len(self.ts_allvars) > 0:
            if saved_name is None:
                saved_name = self._tfObject__name
            ts_dict = {}
            for ts in self.ts_allvars:
                splits = ts.name.split('/')
                # XXX only restore the vars match the saved_name!!!!
                if splits[0] != saved_name:
                    continue
                splits[0] = saved_name
                saved_ts_name = '/'.join(splits)
                assert saved_ts_name.split(':')[1] == '0'
                saved_ts_name = saved_ts_name.split(':')[0]
                ts_dict[saved_ts_name] = ts
            saver = tf.train.Saver(ts_dict)
            saver.restore(tf.get_default_session(), path)


"""
Session management.
"""


def make_session(num_cpu=None, make_default=False):
    """Returns a session that will use <num_cpu> CPU's only"""
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    tf_config.gpu_options.allocator_type = 'BFC'
    tf_config.gpu_options.allow_growth = True
    if make_default:
        return tf.InteractiveSession(config=tf_config)
    else:
        return tf.Session(config=tf_config)


def single_threaded_session():
    """Returns a session which will only use a single CPU"""
    return make_session(num_cpu=1)


"""
Placeholder cache. Create if necessary.
"""
_PLACEHOLDER_CACHE = {}  # name -> (placeholder, dtype, shape)


def get_placeholder(name, dtype=None, shape=None):
    if name in _PLACEHOLDER_CACHE:
        assert dtype is None
        assert shape is None
        return _PLACEHOLDER_CACHE[name][0]
    else:
        out = tf.placeholder(dtype=dtype, shape=shape, name=name)
        _PLACEHOLDER_CACHE[name] = (out, dtype, shape)
        return out


"""
Simple functions that construct tensors from tensors.
"""


def squared_sum(sy_x, axis=None):
    sy_x_sqr = tf.square(sy_x)
    return tf.reduce_sum(sy_x_sqr, axis=axis)


def switch(condition, then_expression, else_expression):
    """Switches between two operations depending on a scalar value (int or bool).
    Note that both `then_expression` and `else_expression`
    should be symbolic tensors of the *same shape*.

    Args:
      condition: scalar tensor.
      then_expression: TensorFlow operation.
      else_expression: TensorFlow operation.
    """
    sy_x_shape = copy.copy(then_expression.get_shape())
    sy_x = tf.cond(tf.cast(condition, 'bool'),
                   lambda: then_expression,
                   lambda: else_expression)
    sy_x.set_shape(sy_x_shape)
    return sy_x


def build_multilayer_perceptron(
        scope,
        sy_input,
        output_size,
        n_layers=2,
        size=64,
        activation=tf.tanh,
        hid_layer_std=1.0,
        output_activation=None,
        output_init_std=1.0,
        reuse=False,
):

    with tf.variable_scope(scope, reuse=reuse):
        sy_y = sy_input
        for _ in range(n_layers):
            sy_y = tf.layers.dense(sy_y, size, activation=activation,
                                   kernel_initializer=normc_initializer(hid_layer_std))
        sy_y = tf.layers.dense(sy_y, output_size, activation=output_activation,
                               kernel_initializer=normc_initializer(output_init_std))
    return sy_y


def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def build_get(tensors):
    return function([], tensors)


def build_set(tensors):
    def get_ph(x, name=None):
        if type(x) is not list:
            return [tf.placeholder(shape=x.shape, dtype=x.dtype)]
        else:
            return [tf.placeholder(shape=v.shape, dtype=v.dtype) for v in x]
    phs = get_ph(tensors)
    assign_ops = [tf.assign(t, p) for (t, p) in zip(tensors, phs)]
    set_fun = function(phs, [], assign_ops)
    return set_fun


"""
Convert from flat tensors to list of tensors and back.
"""

"""
Shape related.
"""


def get_shape(x):
    return x.get_shape().as_list()


def intprod(shape):
    return int(np.prod(shape))


def get_size(x):
    shape = get_shape(x)
    assert all(isinstance(a, int) for a in shape), "shape function assumes that shape is fully known"
    return intprod(shape)


def get_tensor_by_name(name):
    return tf.get_default_graph().get_tensor_by_name('{}:0'.format(name))


class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32):
        assigns = []
        shapes = list(map(get_shape, var_list))
        total_size = np.sum([intprod(shape) for shape in shapes])

        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = intprod(shape)
            assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        tf.get_default_session().run(self.op, feed_dict={self.theta: theta})


class GetFlat(object):
    def __init__(self, var_list):
        self.op = tf.concat(axis=0, values=[tf.reshape(v, [get_size(v)]) for v in var_list])

    def __call__(self):
        return tf.get_default_session().run(self.op)


class Shaper(object):
    """
        A wrapper of a list of tf.Tensors for convenient conversions between a
        list of tensors and its flat counterpart based on contiguous memory
        allocation.

        It creates tensorflow operators only when necessary.
    """

    def __init__(self, tensors):
        # tensors: a list of tensors.
        self._tensors = tensors

    @property
    def _tensor_shapes(self):
        return [t.shape.as_list() for t in self._tensors]

    # for np.ndarray
    def unflatten(self, val):
        return unflatten(val, shapes=self._tensor_shapes)

    def flatten(self, vs):
        return flatten(vs)

    def build_flat_ph(self):
        """ Create return a single placeholder of the size as the number of
        elements in self.tensors. Return the placeholder and a list of
        tf.Tensors view of the created placeholder in accordinance with the
        structure of self.tensors. """

        total_size = sum([intprod(shape) for shape in self._tensor_shapes])
        ph = tf.placeholder(dtype=tf.float32, shape=[total_size])
        idx = 0
        vs = []
        for shape in self._tensor_shapes:
            size = intprod(shape)
            vs.append(tf.reshape(ph[idx:idx + size], shape))
            idx += size
        return ph, vs

    @property
    def variables(self):
        if not hasattr(self, '_get_variables'):
            self._get_variables = build_get(self._tensors)
        return self._get_variables()

    @variables.setter
    def variables(self, vals):
        if not hasattr(self, '_set_variables'):
            self._set_variables = build_set(self._tensors)
        self._set_variables(*vals)

    @property
    def variable(self):
        return self.flatten(self.variables)

    @variable.setter
    def variable(self, val):
        self.variables = self.unflatten(val)


"""


Create callable functions from tensors.
"""


def function(inputs, outputs, updates=None, givens=None):
    """Just like Theano function. Take a bunch of tensorflow placeholders and expressions
    computed based on those placeholders and produces f(inputs) -> outputs. Function f takes
    values to be fed to the input's placeholders and produces the values of the expressions
    in outputs.

    Input values can be passed in the same order as inputs or can be provided as kwargs based
    on placeholder name(passed to constructor or accessible via placeholder.op.name).

    Example:
        x = tf.placeholder(tf.int32, (), name="x")
        y = tf.placeholder(tf.int32, (), name="y")
        z = 3 * x + 2 * y
        lin = function([x, y], z, givens={y: 0})

        with single_threaded_session():
            initialize()

            assert lin(2) == 6
            assert lin(x=3) == 9
            assert lin(2, 2) == 10
            assert lin(x=2, y=3) == 12

    Parameters
    ----------
    inputs: [tf.placeholder, tf.constant, or object with make_feed_dict method]
        list of input arguments
    outputs: [tf.Variable] or tf.Variable
        list of outputs or a single output to be returned from function. Returned
        value will also have the same shape.
    """
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), f(*args, **kwargs)))
    else:
        f = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: f(*args, **kwargs)[0]


class _Function(object):
    def __init__(self, inputs, outputs, updates, givens):
        for inpt in inputs:
            if not hasattr(inpt, 'make_feed_dict') and not (type(inpt) is tf.Tensor and len(inpt.op.inputs) == 0):
                assert False, "inputs should all be placeholders, constants, or have a make_feed_dict method"
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens

    def _feed_input(self, feed_dict, inpt, value):
        if hasattr(inpt, 'make_feed_dict'):
            feed_dict.update(inpt.make_feed_dict(value))
        else:
            feed_dict[inpt] = value

    def __call__(self, *args):
        assert len(args) <= len(self.inputs), "Too many arguments provided"
        feed_dict = {}
        # Update the args
        for inpt, value in zip(self.inputs, args):
            self._feed_input(feed_dict, inpt, value)
        # Update feed dict with givens.
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        results = tf.get_default_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]
        return results
