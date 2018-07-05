
"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

#import functions

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 224
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
INPUT_IMAGE_SIZE = 128
INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128
INPUT_IMAGE_CHANNELS = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 15782 # CASIA2
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1950


# 128 patch dataset
IMAGE_SIZE = 128
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
INPUT_IMAGE_SIZE = 128
INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128
INPUT_IMAGE_CHANNELS = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 25782 # CASIA2
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 3102


# Number of classes
NUM_CLASSES = 2
# This is the number of training examples in the dataset - one epoch runs over all the examples


def read_dataset(filename_queue):
  """Reads and parses examples from data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class datasetRecord(object):
    pass
  result = datasetRecord()

  # 1 byte label followed by all the pixels (INPUT_IMAGE_CHANNELS, INPUT_IMAGE_WIDTH by INPUT_IMAGE_HEIGHT
  label_bytes = 1
  result.height = INPUT_IMAGE_WIDTH
  result.width = INPUT_IMAGE_HEIGHT
  result.depth = INPUT_IMAGE_CHANNELS
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in this format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  #print("The value from reading: {} \n".format(value))

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)
  
  #print(record_bytes)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The STL labels run from 1-10 will this make a difference? Better to normalise just in case...
  #result.label = result.label - 1
  #print("Here is the label: {}, hooray!".format(result.label))

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                           [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result



def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, INPUT_IMAGE_CHANNELS] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  #num_preprocess_threads = 16
  num_preprocess_threads = 1
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + INPUT_IMAGE_CHANNELS * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + INPUT_IMAGE_CHANNELS * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size, distort=2):
  """Construct distorted input for training using the Reader ops.

  Args:
    data_dir: Path to the dataset data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, INPUT_IMAGE_CHANNELS] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # for CIFAR-10
  print("From within distorted_inputs, data_dir = {}here".format(data_dir))
  #filenames = [os.path.join(data_dir, 'patches_%d.bin' % i) for i in xrange(0, 8)]
  #filenames = [os.path.join(data_dir, 'patches_train_%d.bin' % i) for i in xrange(0, 1)]
  #filenames = [os.path.join(data_dir, 'train_crop.bin')]
  filenames = [os.path.join(data_dir, 'train_crop_%d.bin' % i) for i in xrange(0, 11)]

  print("Expected filenames: {}".format(filenames))

  myfilenames = []
  for f in filenames:
    if tf.gfile.Exists(f):
      myfilenames.append(f)

  print("Found filenames: {}".format(myfilenames))

  filenames = myfilenames
  if len(filenames) == 0:
    raise ValueError('Failed to find any files to process')

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_dataset(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_HEIGHT
  width = IMAGE_WIDTH

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  if distort == 1:
    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, INPUT_IMAGE_CHANNELS])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
  elif distort == 2:
    distorted_image = tf.random_crop(reshaped_image, [height, width, INPUT_IMAGE_CHANNELS])

  else:
    distorted_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, width, height)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(distorted_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
  print ('Filling queue with %d images before starting to train. ' 'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)



def inputs(eval_data, data_dir, batch_size):
  """Construct input for dataset evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, INPUT_IMAGE_CHANNELS] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    #filenames = [os.path.join(data_dir, 'patches_%d.bin' % i) for i in xrange(0, 8)] # original based on CIF vid
    filenames = [os.path.join(data_dir, 'patches_train_%d.bin' % i) for i in xrange(0, 1)]
    filenames = [os.path.join(data_dir, 'train_crop.bin')]
    # For CASIA2 256x256 patches and 128x128 patches
    filenames = [os.path.join(data_dir, 'train_crop_%d.bin' % i) for i in xrange(0, 11)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    filenames = [os.path.join(data_dir, 'patches_test_%d.bin' % i) for i in xrange(0, 1)]
    filenames = [os.path.join(data_dir, 'test_crop.bin')]
    #filenames = [os.path.join(data_dir, 'test_crop_%d.bin' % i) for i in xrange(0, 10)]
    print("Inputting test data which is {}".format(filenames))


  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_dataset(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_HEIGHT
  width = IMAGE_WIDTH

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, width, height)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(resized_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)
