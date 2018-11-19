
from __future__ import print_function

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import itNet_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
#tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('batch_size', 64, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'qpDataset', """Path to the directory.""")
tf.app.flags.DEFINE_string('batches_dir', ' ', """Path to the secondary data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('binarise_label', 0, """Binarise this label""")
tf.app.flags.DEFINE_integer('training', 1, """Training cycle""")

#tf.app.flags.DEFINE_string('prelearned_checkpoint', '/Users/pam/Documents/data/CIFAR-10/test3/cifar10_train/train_yuv/model.ckpt-29999', """The same network architecture trained on something else""")



# Global constants describing the data set.
IMAGE_SIZE = itNet_input.IMAGE_SIZE
INPUT_IMAGE_CHANNELS = itNet_input.INPUT_IMAGE_CHANNELS
NUM_CLASSES = itNet_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = itNet_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = itNet_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
#NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.5  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# For Nao and Ri
#NUM_EPOCHS_PER_DECAY = 10.0      # Epochs after which learning rate decays.
#LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
#INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_on_cpu_with_constant(name, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay_orig(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd, fresh_init = True, init_tensor=0, verbose=False):
    """Helper to create an initialized Variable with weight decay.
        
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.
        
        Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
        
        Returns:
        Variable Tensor
        """
    if verbose:
        print("The name of the variable: {}".format(name))
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32


    if fresh_init:
        var = _variable_on_cpu(
                   name,
                   shape,
                   tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    else:
        var = _variable_on_cpu_with_constant(
                   name,
                   init_tensor)

    if verbose:
        print("Here's the variable of name {}:".format(name))
        my_vars = tf.Print(var, [var], message="This is var: ")
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def binariseTheLabels(labels):
    print("All right, binarising")
    if (FLAGS.binarise_label >0 ):
        masky = tf.fill(labels.get_shape(), (FLAGS.binarise_label-1))
        labels = tf.equal(labels, masky)
    elif (FLAGS.binarise_label == -2):
        # combine labels (0,1), (2,3), (3,4) etc
        # so label = l/2 rounded down
        divvy = tf.fill(labels.get_shape(), 2)
        labels = tf.floordiv(labels, divvy)
    elif (FLAGS.binarise_label == -3):
        divvy = tf.fill(labels.get_shape(), 3)
        labels = tf.floordiv(labels, divvy)
    else:
        print("But not actually doing anything about the binarising")

    labels = tf.cast(labels, tf.int32)
    return labels



def distorted_inputs():
  """Construct distorted input for training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = FLAGS.data_dir
  if FLAGS.batches_dir.strip():
    print("putting on the batches")
    data_dir = os.path.join(FLAGS.data_dir, FLAGS.batches_dir)

  print("The data dir is {} in distorted_inputs".format(data_dir))
  #images, labels = itNet_input.distorted_inputs(data_dir=data_dir, batch_size=FLAGS.batch_size, distort=False)
  images, labels = itNet_input.distorted_inputs(data_dir=data_dir, batch_size=FLAGS.batch_size, distort=1)

  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)

  # binaries the labels if necessary:
  #print("Binarising here")
  #labels = binariseTheLabels(labels)

  return images, labels


def inputs(eval_data, singleThreaded=False, filename="", numExamplesToTest=NUM_EXAMPLES_PER_EPOCH_FOR_EVAL):
  """Construct input for evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, INPUT_IMAGE_CHANNELS] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = FLAGS.data_dir

  if FLAGS.batches_dir.strip():
    data_dir = os.path.join(FLAGS.data_dir, FLAGS.batches_dir)
  print("The data dir is {} here".format(data_dir))

  images, labels = itNet_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size,
                                        singleThreaded = singleThreaded,
                                        filename = filename,
                                        numExamplesToTest = numExamplesToTest
                                      )
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)

  # binaries the labels if necessary:
  #print("Binarising there")
  #labels = binariseTheLabels(labels)
  return images, labels


def inference_switch(images, type=1):
    #NUM_CLASSES = itNet_input.NUM_CLASSES

    #if FLAGS.binarise_label > 0:
    #    NUM_CLASSES = 2
    #    itNet_input.NUM_CLASSES = 2
    #elif FLAGS.binarise_label == -2:
    #    NUM_CLASSES = 4
    #    itNet_input.NUM_CLASSES = 4
    #elif FLAGS.binarise_label == -3:
    #    NUM_CLASSES = 3
    #    itNet_input.NUM_CLASSES = 3


    print("inference_switch: {}".format(type))

    if type == 1:
        return inference(images)
    elif type == 2:
        return inference_2(images)
    elif type == 3:
        return inference_3(images)
    elif type == 4:
        return inference_4(images, FLAGS.prelearned_checkpoint)
    elif type == 5:
        return inference_5(images)
    elif type == 6:
        return inference_6(images)
    elif type == 7:
        return inference_7(images)
    elif type == 8:
        return inference_8(images)
    elif type == 9:
        return inference_9(images)
    elif type == 10:
        return inference_10(images)
    elif type == 11:
        return inference_11(images)
    elif type == 12:
        return inference_12(images)
    elif type == 13:
        return inference_13(images)
    elif type == 14:
        return inference_14(images)
    elif type == 15:
        return inference_15(images)
    elif type == 16:
        return inference_16(images)
    elif type == 17:
        return inference_17(images)
    elif type == 18:
        return inference_18(images)
    elif type == 19:
        return inference_19(images)
    elif type == 20:
        return inference_10(images, 0.5)
    elif type == 21:
        return inference_10(images, 0.8)
    elif type == 22:
        return inference_10(images, 0.2)
    elif type == 23:
        dropout_rate = 0.8
        if FLAGS.training:
            dropout_rate = 1.0
        return inference_23(images, dropout_rate)
    elif type == 24:
        return inference_24(images)
    elif type == 25:
        dropout_rate = 0.5
        if FLAGS.training:
            dropout_rate = 1.0
        return inference_25(images, dropout_rate)
    elif type == 26:
        dropout_rate = 0.8
        if FLAGS.training:
            dropout_rate = 1.0
        return inference_25(images, dropout_rate)
    elif type == 27:
        dropout_rate = 0.3
        if FLAGS.training:
            dropout_rate = 1.0
        return inference_25(images, dropout_rate)
    elif type == 28:
        dropout_rate = 0.5
        if FLAGS.training:
            dropout_rate = 1.0
        return inference_28(images, dropout_rate)


def inference(images):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # PAJ: This is the original from the CIFAR-10 tutorial. Expect 84% accuracy after 30k steps on CIFAR-10

  # conv1
  with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, INPUT_IMAGE_CHANNELS, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

  # conv2
  with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3', reuse=tf.AUTO_REUSE) as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4', reuse=tf.AUTO_REUSE) as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear', reuse=tf.AUTO_REUSE) as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def inference_2(images):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.

  Notes:
      It's a different network from the original one, see how it goes....
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # PAJ: This is Pam's version of AlexNet for CIFAR-10. Same number of conv and fc layers.
  # It converged on CIFAR-10, 64x64 but it's accuracy was 10% (so it didn't learn anything!).
  # Probably need to work out how to add drop out to this.
  # conv1
  with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[7, 7, INPUT_IMAGE_CHANNELS, 96],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)
  # norm1
  norm1 = tf.nn.lrn(conv1, 2, bias=1.0, alpha=2e-05 , beta=0.75, name='norm1')

  # pool1
  pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

  # conv2
  with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 96, 256],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 2, bias=1.0, alpha=2e-05, beta=0.75, name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')


  # conv3
  with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 384],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3)

  # conv4
  with tf.variable_scope('conv4', reuse=tf.AUTO_REUSE) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 384, 384],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv4)

  # conv5
  with tf.variable_scope('conv5', reuse=tf.AUTO_REUSE) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 384, 256],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv5)

  #pool5
  pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')

  # fc6
  with tf.variable_scope('fc6', reuse=tf.AUTO_REUSE) as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool5, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 512],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
    fc6 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(fc6)

  # fc7
  with tf.variable_scope('fc7', reuse=tf.AUTO_REUSE) as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(fc6, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 256],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    fc7 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(fc7)

  # fc8
  with tf.variable_scope('fc8', reuse=tf.AUTO_REUSE) as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(fc7, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 10],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [10], tf.constant_initializer(0.1))
    fc8 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(fc8)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear', reuse=tf.AUTO_REUSE) as scope:
    weights = _variable_with_weight_decay('weights', [10, NUM_CLASSES],
                                          stddev=1/10.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(fc8, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

# from https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412
def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        _activation_summary(beta)
        _activation_summary(gamma)
        _activation_summary(batch_mean)
        _activation_summary(batch_var)

        ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])

            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(tf.cast(phase_train, tf.bool),
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        #tf.summary.scalar(ema_apply_op.name, ema_apply_op.average(0))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        _activation_summary(normed)
    return normed

def inference_23(images, dropOut_prob = 1.0, batchNorm = False):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.

  Notes:
      It's a different network from the original one, see how it goes....
  """
  # PAJ: This is Pam's version of AlexNet for tampering detection. Same number of conv and fc layers.
  # Using CASIA-2, it converged to 85% accuracy but it takes about 50k steps and sometimes it
  # simply sticks at loss = 0.7 (that's inference_2 above)
  # This version adds in batch normalisation and takes out biases because Andrew Ng says you can.
  # conv1
  with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[7, 7, INPUT_IMAGE_CHANNELS, 96],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')

    # batch norm
    # Andrew Ng says you don't need the biases...?
    # Here https://www.coursera.org/learn/deep-neural-network/lecture/RN8bN/fitting-batch-norm-into-a-neural-network
    if batchNorm:
        normed = batch_norm(conv, 96, FLAGS.training)
        conv1 = tf.nn.relu(normed, name=scope.name)
    else:
        biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)
  # norm1
  norm1 = tf.nn.lrn(conv1, 2, bias=1.0, alpha=2e-05 , beta=0.75, name='norm1')

  # pool1
  pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

  # conv2
  with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 96, 256],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    if batchNorm:
        # batch norm
        normed = batch_norm(conv, 256, FLAGS.training)
        conv2 = tf.nn.relu(normed, name=scope.name)
    else:
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 2, bias=1.0, alpha=2e-05, beta=0.75, name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')


  # conv3
  with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 384],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    # batch norm
    if batchNorm:
        normed = batch_norm(conv, 384, FLAGS.training)
        conv3 = tf.nn.relu(normed, name=scope.name)
    else:
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3)

  # conv4
  with tf.variable_scope('conv4', reuse=tf.AUTO_REUSE) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 384, 384],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    # batch norm
    if batchNorm:
        normed = batch_norm(conv, 384, FLAGS.training)
        conv4 = tf.nn.relu(normed, name=scope.name)
    else:
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv4)

  # conv5
  with tf.variable_scope('conv5', reuse=tf.AUTO_REUSE) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 384, 256],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    if batchNorm:
        # batch norm
        normed = batch_norm(conv, 256, FLAGS.training)
        conv5 = tf.nn.relu(normed, name=scope.name)
    else:
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv5)

  #pool5
  pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')

  # fc6
  with tf.variable_scope('fc6', reuse=tf.AUTO_REUSE) as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool5, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 512],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
    relu6 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    fc6 = tf.nn.dropout(relu6, dropOut_prob)
    _activation_summary(fc6)

  # fc7
  with tf.variable_scope('fc7', reuse=tf.AUTO_REUSE) as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(fc6, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 256],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    relu7 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    fc7 = tf.nn.dropout(relu7, dropOut_prob)
    _activation_summary(fc7)

  # fc8
  with tf.variable_scope('fc8', reuse=tf.AUTO_REUSE) as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(fc7, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 10],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [10], tf.constant_initializer(0.1))
    relu8 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    fc8 = tf.nn.dropout(relu8, dropOut_prob)
    _activation_summary(fc8)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear', reuse=tf.AUTO_REUSE) as scope:
    weights = _variable_with_weight_decay('weights', [10, NUM_CLASSES],
                                          stddev=1/10.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(fc8, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def inference_24(images):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # PAJ: A short network with batch normalization, but I can't make it work/save batch norm weights.

  # conv1
  with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, INPUT_IMAGE_CHANNELS, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    # batch norm
    normed = batch_norm(conv, 64, FLAGS.training)
    conv1 = tf.nn.relu(normed, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

  # conv2
  with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    # batch norm
    normed = batch_norm(conv, 64, FLAGS.training)
    conv2 = tf.nn.relu(normed, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3', reuse=tf.AUTO_REUSE) as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4', reuse=tf.AUTO_REUSE) as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear', reuse=tf.AUTO_REUSE) as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear





def inference_3(images):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, INPUT_IMAGE_CHANNELS, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # pool2
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2')
  # norm2
  norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')

  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3)

  # norm3
  norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm3')
  # pool2
  pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool3')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool3, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear


def inference_4(images, input):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.

  Notes:
      It's a different network from the original one, this one takes 64x64 images
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # PAJ: This one designed to lift its initial weights from some checkpoint file specified in the flags (I think!)

  reader = tf.train.NewCheckpointReader(FLAGS.prelearned_checkpoint)
  full_name = 'conv1/weights'
  var = reader.get_tensor(full_name)
  var_c1_w = tf.pack(var)
  full_name = 'conv1/biases'
  var = reader.get_tensor(full_name)
  var_c1_b = tf.pack(var)
  full_name = 'conv2/weights'
  var = reader.get_tensor(full_name)
  var_c2_w = tf.pack(var)
  full_name = 'conv2/biases'
  var = reader.get_tensor(full_name)
  var_c2_b = tf.pack(var)
  full_name = 'local3/weights'
  var = reader.get_tensor(full_name)
  var_l3_w = tf.pack(var)
  full_name = 'local3/biases'
  var = reader.get_tensor(full_name)
  var_l3_b = tf.pack(var)
  full_name = 'local4/weights'
  var = reader.get_tensor(full_name)
  var_l4_w = tf.pack(var)
  full_name = 'local4/biases'
  var = reader.get_tensor(full_name)
  var_l4_b = tf.pack(var)
  full_name = 'softmax_linear/weights'
  var = reader.get_tensor(full_name)
  var_softmax_w = tf.pack(var)
  full_name = 'softmax_linear/biases'
  var = reader.get_tensor(full_name)
  var_softmax_b = tf.pack(var)


  # conv1
  with tf.variable_scope('conv1') as scope:
    print("Initialising the network with kernels")
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, INPUT_IMAGE_CHANNELS, 64],
                                         stddev=5e-2,
                                         wd=0.0,
                                         verbose=True,
                                         fresh_init=False,
                                         init_tensor=var_c1_w)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu_with_constant('biases', var_c1_b)
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0,
                                         fresh_init=False,
                                         init_tensor=var_c2_w)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu_with_constant('biases', var_c2_b)
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights',
                                          shape=[dim, 384],
                                          stddev=0.04,
                                          wd=0.004,
                                          fresh_init=False,
                                          init_tensor=var_l3_w)
    biases = _variable_on_cpu_with_constant('biases',var_l3_b)
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:

    weights = _variable_with_weight_decay('weights',
                                          shape=[384, 192],
                                          stddev=0.04,
                                          wd=0.004,
                                          fresh_init = False,
                                          init_tensor = var_l4_w)

    biases = _variable_on_cpu_with_constant('biases', var_l4_b)
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:

    weights = _variable_with_weight_decay('weights',
                                          [192, NUM_CLASSES],
                                          stddev=1/192.0,
                                          wd=0.0,
                                          fresh_init = False,
                                          init_tensor = var_softmax_w)

    biases = _variable_on_cpu_with_constant('biases', var_softmax_b)
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def inference_5(images):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # PAJ: Inspired by Simonyan and Zisserman, replaced 5x5 with 2x 3x3 conv layers.

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, INPUT_IMAGE_CHANNELS, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 128],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')
  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 256],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3)

  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 256],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv4)

  # pool4
  pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

  # local5
  with tf.variable_scope('local5') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool4, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local5 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local5)

  # local6
  with tf.variable_scope('local6') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local6 = tf.nn.relu(tf.matmul(local5, weights) + biases, name=scope.name)
    _activation_summary(local6)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES], stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local6, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def inference_6(images):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # PAJ: This is the original from the tutorial. Expect 84% accuracy after 30k steps on CIFAR-10

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, INPUT_IMAGE_CHANNELS, 160],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [160], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 160, 160],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [160], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def inference_7(images):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # PAJ: This is the original from the CIFAR-10 tutorial. Expect 84% accuracy after 30k steps on CIFAR-10

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[9, 9, INPUT_IMAGE_CHANNELS, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 8, 8, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def inference_8(images):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # PAJ: This is the original from the CIFAR-10 tutorial. Expect 84% accuracy after 30k steps on CIFAR-10

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[4, 4, INPUT_IMAGE_CHANNELS, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[4, 4, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def inference_9(images):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # PAJ: This is the original from the CIFAR-10 tutorial. Expect 84% accuracy after 30k steps on CIFAR-10

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[18, 18, INPUT_IMAGE_CHANNELS, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 16, 16, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[9, 9, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def inference_10(images, dropOut_prob = 1.0):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # PAJ: This is the original from the CIFAR-10 tutorial. Expect 84% accuracy after 30k steps on CIFAR-10

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, INPUT_IMAGE_CHANNELS, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    relu3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    local3 = tf.nn.dropout(relu3, dropOut_prob)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    relu4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    local4 = tf.nn.dropout(relu4, dropOut_prob)
    _activation_summary(local4)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def inference_11(images):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # PAJ: Inspired by Simonyan and Zisserman, replaced 5x5 with 2x 3x3 conv layers.

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[4, 4, INPUT_IMAGE_CHANNELS, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[4, 4, 64, 128],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')
  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[4, 4, 128, 256],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(pool2, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3)

  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[4, 4, 256, 256],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv3, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv4)

  # pool4
  pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

  # local5
  with tf.variable_scope('local5') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool4, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local5 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local5)

  # local6
  with tf.variable_scope('local6') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local6 = tf.nn.relu(tf.matmul(local5, weights) + biases, name=scope.name)
    _activation_summary(local6)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES], stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local6, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear


def inference_12(images):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # PAJ: Inspired by Simonyan and Zisserman, replaced 5x5 with 2x 3x3 conv layers.

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[4, 4, INPUT_IMAGE_CHANNELS, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[4, 4, 64, 128],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(pool1, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  #norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

  # pool2
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')
  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[4, 4, 128, 256],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(pool2, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3)

  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[4, 4, 256, 256],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv3, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv4)

  # pool4
  pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

  # local5
  with tf.variable_scope('local5') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool4, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local5 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local5)

  # local6
  with tf.variable_scope('local6') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local6 = tf.nn.relu(tf.matmul(local5, weights) + biases, name=scope.name)
    _activation_summary(local6)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES], stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local6, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def inference_13(images):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # PAJ: Inspired by Simonyan and Zisserman, replaced 5x5 with 2x 3x3 conv layers.

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[4, 4, INPUT_IMAGE_CHANNELS, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[4, 4, 64, 128],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')
  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[4, 4, 128, 256],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3)

  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[4, 4, 256, 256],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv4)

  # pool4
  pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

  # local5
  with tf.variable_scope('local5') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool4, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local5 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local5)

  # local6
  with tf.variable_scope('local6') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local6 = tf.nn.relu(tf.matmul(local5, weights) + biases, name=scope.name)
    _activation_summary(local6)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES], stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local6, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def inference_14(images):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # PAJ: This is the original from the CIFAR-10 tutorial. Expect 84% accuracy after 30k steps on CIFAR-10

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, INPUT_IMAGE_CHANNELS, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(pool1, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # pool2
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)


  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local3, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def inference_15(images):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # PAJ: This is the original from the CIFAR-10 tutorial. Expect 84% accuracy after 30k steps on CIFAR-10

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, INPUT_IMAGE_CHANNELS, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(pool1, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # pool2
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3)

  # local4
  with tf.variable_scope('local4') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(conv3, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local4)


  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def inference_16(images):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # PAJ: This is the original from the CIFAR-10 tutorial. Expect 84% accuracy after 30k steps on CIFAR-10

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, INPUT_IMAGE_CHANNELS, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 4, 4, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def inference_17(images):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # PAJ: This is the original from the CIFAR-10 tutorial. Expect 84% accuracy after 30k steps on CIFAR-10

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, INPUT_IMAGE_CHANNELS, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv1, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(conv2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def inference_18(images):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # PAJ: This is the original from the CIFAR-10 tutorial. Expect 84% accuracy after 30k steps on CIFAR-10

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, INPUT_IMAGE_CHANNELS, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(pool2, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3)

  # norm3
  norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
  # pool3
  pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

  # local4
  with tf.variable_scope('local4') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool3, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # local5
  with tf.variable_scope('local5') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local5 = tf.nn.relu(tf.matmul(local4, weights) + biases, name=scope.name)
    _activation_summary(local5)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local5, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def inference_19(images):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """

  # PAJ: Inspired by A DEEP NEURAL NETWORK FOR IMAGE QUALITY ASSESSMENT (ok, it's nearly the same network).

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, INPUT_IMAGE_CHANNELS, 32],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 32, 32],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv1, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)


  # pool2
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 32, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(pool2, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3)

  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv3, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv4)

  # pool4
  pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

  # conv5
  with tf.variable_scope('conv5') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 128],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(pool4, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv5)

  # conv6
  with tf.variable_scope('conv6') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 128],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv5, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv6 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv6)

  # pool6
  pool6 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool6')

  # conv7
  with tf.variable_scope('conv7') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           shape=[3, 3, 128, 256],
                                           stddev=5e-2,
                                           wd=0.0)
      conv = tf.nn.conv2d(pool6, kernel, [1, 2, 2, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      conv7 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv7)

  # conv8
  with tf.variable_scope('conv8') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           shape=[3, 3, 256, 256],
                                           stddev=5e-2,
                                           wd=0.0)
      conv = tf.nn.conv2d(conv7, kernel, [1, 2, 2, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      conv8 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv8)

  # pool8
  pool8 = tf.nn.max_pool(conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool6')

  # conv9
  with tf.variable_scope('conv9') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           shape=[3, 3, 256, 512],
                                           stddev=5e-2,
                                           wd=0.0)
      conv = tf.nn.conv2d(pool8, kernel, [1, 2, 2, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      conv9 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv9)

  # conv10
  with tf.variable_scope('conv10') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           shape=[3, 3, 512, 512],
                                           stddev=5e-2,
                                           wd=0.0)
      conv = tf.nn.conv2d(conv9, kernel, [1, 2, 2, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      conv10 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv10)

  # local11
  with tf.variable_scope('local11') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(conv10, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 512],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
    local11 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local11)

  # local12
  #with tf.variable_scope('local12') as scope:
  #  weights = _variable_with_weight_decay('weights', shape=[512, 192],
  #                                        stddev=0.04, wd=0.004)
  #  biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
  #  local12 = tf.nn.relu(tf.matmul(local11, weights) + biases, name=scope.name)
  #  _activation_summary(local12)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [512, NUM_CLASSES],
                                          stddev=1/512.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local11, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def inference_25(images, dropOut_prob = 1.0):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # PAJ: this is from the paper A deep learning approach to detection of splicing and copy-move forgeries in images
  # By Nao and Ri at WIFS who used CASIA 2 and claim
  # 97 percent but they've randomised their dataset
  # (so train and test are not disjoint)

  weightdecay = 0.0001

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, INPUT_IMAGE_CHANNELS, 30],
                                         stddev=5e-2,
                                         wd=weightdecay)
    conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [30], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # (conv2) layer 2
  # norm1
  norm2 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

  # conv3 because the paper gives the mp layer a number
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 30, 16],
                                         stddev=5e-2,
                                         wd=weightdecay)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3)

  # conv4 because the paper gives the mp layer a number
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 16, 16],
                                         stddev=5e-2,
                                         wd=weightdecay)
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv4)

  # conv5 because the paper gives the mp layer a number
  with tf.variable_scope('conv5') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 16, 16],
                                         stddev=5e-2,
                                         wd=weightdecay)
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv5)

  # (conv6) layer 6
  # norm1
  norm6 = tf.nn.lrn(conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm6')
  # pool2
  pool6 = tf.nn.max_pool(norm6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool6')

  # conv7 because the paper gives the mp layer a number
  with tf.variable_scope('conv7') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 16, 16],
                                         stddev=5e-2,
                                         wd=weightdecay)
    conv = tf.nn.conv2d(pool6, kernel, [1, 1, 1, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv7 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv7)

  # conv8 because the paper gives the mp layer a number
  with tf.variable_scope('conv8') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 16, 16],
                                         stddev=5e-2,
                                         wd=weightdecay)
    conv = tf.nn.conv2d(conv7, kernel, [1, 1, 1, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv8 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv8)

  # conv9 because the paper gives the mp layer a number
  with tf.variable_scope('conv9') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 16, 16],
                                         stddev=5e-2,
                                         wd=weightdecay)
    conv = tf.nn.conv2d(conv8, kernel, [1, 1, 1, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv9 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv9)

  # conv10 because the paper gives the mp layer a number
  with tf.variable_scope('conv10') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 16, 16],
                                         stddev=5e-2,
                                         wd=weightdecay)
    conv = tf.nn.conv2d(conv9, kernel, [1, 1, 1, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv10 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv10)


  # "Output"
  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    reshape = tf.reshape(conv10, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', [dim, NUM_CLASSES],
                                          stddev=1/2.0, wd=weightdecay)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(reshape, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def inference_28(images, dropOut_prob = 1.0):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # PAJ: this is from the paper A deep learning approach to detection of splicing and copy-move forgeries in images
  # By Nao and Ri at WIFS who used CASIA 2 and claim
  # 97 percent but they've randomised their dataset
  # (so train and test are not disjoint)

  weightdecay = 0.0001

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, INPUT_IMAGE_CHANNELS, 30],
                                         stddev=5e-2,
                                         wd=weightdecay)
    conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [30], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # (conv2) layer 2
  # norm1
  norm2 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

  # conv3 because the paper gives the mp layer a number
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 30, 16],
                                         stddev=5e-2,
                                         wd=weightdecay)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3)

  # conv4 because the paper gives the mp layer a number
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 16, 16],
                                         stddev=5e-2,
                                         wd=weightdecay)
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv4)

  # conv5 because the paper gives the mp layer a number
  with tf.variable_scope('conv5') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 16, 16],
                                         stddev=5e-2,
                                         wd=weightdecay)
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv5)

  # (conv6) layer 6
  # norm1
  norm6 = tf.nn.lrn(conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm6')
  # pool2
  pool6 = tf.nn.max_pool(norm6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool6')

  # conv7 because the paper gives the mp layer a number
  with tf.variable_scope('conv7') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 16, 16],
                                         stddev=5e-2,
                                         wd=weightdecay)
    conv = tf.nn.conv2d(pool6, kernel, [1, 1, 1, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv7 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv7)

  # conv8 because the paper gives the mp layer a number
  with tf.variable_scope('conv8') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 16, 16],
                                         stddev=5e-2,
                                         wd=weightdecay)
    conv = tf.nn.conv2d(conv7, kernel, [1, 1, 1, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv8 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv8)

  # "Output"
  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    reshape = tf.reshape(conv8, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', [dim, NUM_CLASSES],
                                          stddev=1/2.0, wd=weightdecay)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(reshape, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name +' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train_orig(total_loss, global_step):
  """Train model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
  #decay_steps = 2000

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def train_adam(total_loss, global_step):
  """Train model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
  #decay_steps = 2000

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    #opt = tf.train.GradientDescentOptimizer(lr)
    opt = tf.train.AdamOptimizer()
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op
