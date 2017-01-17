from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import data_provider
import models

from menpo.visualize import print_progress
from pathlib import Path

from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import variable_scope

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 15, '''The batch size to use.''')
tf.app.flags.DEFINE_string('model', 'audio','''Which model is going to be used: audio,video, or both ''')
tf.app.flags.DEFINE_string('dataset_dir', '/vol/atlas/homes/pt511/db/URTIC/tf_records', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('checkpoint_dir', '/vol/atlas/homes/pt511/ckpt/Interspeech2017/train_original_2/', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('log_dir', '/vol/atlas/homes/pt511/ckpt/Interspeech2017/train_original_2/', 'The tfrecords directory.')
tf.app.flags.DEFINE_integer('num_examples', None, 'The number of examples in the test set')
tf.app.flags.DEFINE_string('eval_interval_secs', 300, 'The number of examples in the test set')
tf.app.flags.DEFINE_string('portion', 'devel', 'The portion of the dataset to use -- `train`, `devel`, or `test`.')

def evaluate(data_folder):
  """Evaluates the audio model.
  
  Args:
     data_folder: The folder that contains the data to evaluate the audio model.
  """
    
  g = tf.Graph()
  with g.as_default():
    # Load dataset.
    audio, labels, num_examples = data_provider.get_split(
        data_folder, FLAGS.portion, FLAGS.batch_size)
    
    # Define model graph.
    with slim.arg_scope([slim.batch_norm],
                           is_training=False):
      predictions = models.get_model(FLAGS.model)(audio)

      pred_argmax = tf.argmax(predictions, 1)
      lab_argmax = tf.argmax(labels, 1)

      names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
          "eval/accuracy": slim.metrics.streaming_accuracy(pred_argmax, lab_argmax),
          "eval/UAR": UAR(pred_argmax, lab_argmax)
      })

      summary_ops = []
      # Create the summary ops such that they also print out to std output.
      for name, value in names_to_values.items():
          op = tf.summary.scalar(name, value)
          op = tf.Print(op, [value], name)
          summary_ops.append(op)

      
      num_examples = FLAGS.num_examples or num_examples
      num_batches = num_examples // FLAGS.batch_size
      logging.set_verbosity(1)
      
      # Setup the global step.
      slim.get_or_create_global_step()
    
      # How often to run the evaluation.
      eval_interval_secs = FLAGS.eval_interval_secs 
        
      slim.evaluation.evaluation_loop(
          '',
          FLAGS.checkpoint_dir,
          FLAGS.log_dir,
          num_evals=num_batches,
          eval_op=list(names_to_updates.values()),
          summary_op=tf.merge_summary(summary_ops),
          eval_interval_secs=eval_interval_secs)

def UAR(labels, predictions):

  predictions.get_shape().assert_is_compatible_with(labels.get_shape())

  if labels.dtype != predictions.dtype:
    predictions = math_ops.cast(predictions, labels.dtype)
  
  lab_1 = tf.equal(labels, tf.ones_like(labels))
  lab_0 = tf.equal(labels, tf.zeros_like(labels))

  pred_1 = tf.equal(predictions, tf.ones_like(predictions))
  pred_0 = tf.equal(predictions, tf.zeros_like(predictions))

  tp1 = tf.reduce_sum(tf.cast(tf.logical_and(lab_1, pred_1), tf.float32))
  fn1 = tf.reduce_sum(tf.cast(tf.logical_and(lab_1, pred_0), tf.float32))

  tp2 = tf.reduce_sum(tf.cast(tf.logical_and(lab_0, pred_0), tf.float32))
  fn2 = tf.reduce_sum(tf.cast(tf.logical_and(lab_0, pred_1), tf.float32))

  recall_1 = tp1/tf.maximum((tp1+fn1), tf.constant(0.00000000001, dtype=tf.float32))
  recall_2 = tp2/tf.maximum((tp2+fn2), tf.constant(0.00000000001, dtype=tf.float32))
   
  return mean([recall_1, recall_2], 'UAR')

def mean(values, name=None):
  with variable_scope.variable_scope(name, 'mean', (values)):
    values = math_ops.to_float(values)

    total = _create_local('total', shape=[])
    count = _create_local('count', shape=[])

    num_values = math_ops.to_float(array_ops.size(values))

    update_total_op = state_ops.assign_add(total, math_ops.reduce_sum(values))
    update_count_op = state_ops.assign_add(count, num_values)

    mean_t = _safe_div(total, count, 'value')
    update_op = _safe_div(update_total_op, update_count_op, 'update_op')

    return mean_t, update_op

def _safe_div(numerator, denominator, name):
  """Divides two values, returning 0 if the denominator is <= 0.
  Args:
    numerator: A real `Tensor`.
    denominator: A real `Tensor`, with dtype matching `numerator`.
    name: Name for the returned op.
  Returns:
    0 if `denominator` <= 0, else `numerator` / `denominator`
  """
  return array_ops.where(
      math_ops.greater(denominator, 0),
      math_ops.truediv(numerator, denominator),
      0,
      name=name)

def _create_local(name, shape, collections=None, validate_shape=True,
                  dtype=dtypes.float32):
  """Creates a new local variable.
  Args:
    name: The name of the new or existing variable.
    shape: Shape of the new or existing variable.
    collections: A list of collection names to which the Variable will be added.
    validate_shape: Whether to validate the shape of the variable.
    dtype: Data type of the variables.
  Returns:
    The created variable.
  """
  # Make sure local variables are added to tf.GraphKeys.LOCAL_VARIABLES
  collections = list(collections or [])
  collections += [ops.GraphKeys.LOCAL_VARIABLES]
  return variables.Variable(
      initial_value=array_ops.zeros(shape, dtype=dtype),
      name=name,
      trainable=False,
      collections=collections,
      validate_shape=validate_shape)

def main(_):
    evaluate(FLAGS.dataset_dir)

if __name__ == '__main__':
    tf.app.run()
