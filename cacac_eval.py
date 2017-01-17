from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import data_provider
import models

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
from tensorflow.python.ops import check_ops

from menpo.visualize import print_progress
from pathlib import Path

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 15, '''The batch size to use.''')
tf.app.flags.DEFINE_string('model', 'audio','''Which model is going to be used: audio,video, or both ''')
tf.app.flags.DEFINE_string('dataset_dir', '/vol/atlas/homes/pt511/db/URTIC/tf_records', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('checkpoint_dir', '/vol/atlas/homes/pt511/ckpt/Interspeech2017/train_original_2/', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('log_dir', '/vol/atlas/homes/pt511/ckpt/Interspeech2017/test/', 'The tfrecords directory.')
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
          "eval/UAR": slim.metrics.streaming_mean(UAR(pred_argmax, lab_argmax))
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

  lab_1 = tf.equal(labels, tf.ones_like(labels))
  lab_0 = tf.equal(labels, tf.zeros_like(labels))

  pred_1 = tf.equal(predictions, tf.ones_like(predictions))
  pred_0 = tf.equal(predictions, tf.zeros_like(predictions))

  true_p,  true_positives_update_op = _count_condition(tf.logical_and(lab_1, pred_1))
  false_n, false_negatives_update_op = _count_condition(tf.logical_and(lab_1, pred_0))

  true_n, true_negatives_update_op = _count_condition(tf.logical_and(lab_0, pred_0))
  false_p, false_positives_update_op = _count_condition(tf.logical_and(lab_0, pred_1))

  def compute_recall(true_p, false_n, name):
    return array_ops.where(
        math_ops.greater(true_p + false_n, 0),
        math_ops.div(true_p, true_p + false_n),
        0,
        name)

  recall_1 = compute_recall(true_p, false_n, 'value_1')
  recall_1_update_op = compute_recall(true_positives_update_op, 
                 false_negatives_update_op, 'update_op_1')

  recall_2 = compute_recall(true_n, false_p, 'value_2')
  recall_2_update_op = compute_recall(true_negatives_update_op, 
                 false_positives_update_op, 'update_op_2')

  mean_value = _safe_div(recall_1+recall_2, tf.cast(2, tf.float32), 'value')
  mean_update_op = _safe_div(recall_1_update_op+recall_2_update_op, tf.cast(2, tf.float32),'update_op')

  return mean_value, mean_update_op

def _safe_div(numerator, denominator, name):
  return array_ops.where(
      math_ops.greater(denominator, 0),
      math_ops.truediv(numerator, denominator),
      0,
      name=name)

def _create_local(name, shape, collections=None, validate_shape=True,
                  dtype=dtypes.float32):
  # Make sure local variables are added to tf.GraphKeys.LOCAL_VARIABLES
  collections = list(collections or [])
  collections += [ops.GraphKeys.LOCAL_VARIABLES]
  return variables.Variable(
      initial_value=array_ops.zeros(shape, dtype=dtype),
      name=name,
      trainable=False,
      collections=collections,
      validate_shape=validate_shape)

def _count_condition(values, weights=None, metrics_collections=None,
                     updates_collections=None):
  check_ops.assert_type(values, dtypes.bool)
  count = _create_local('count', shape=[])

  values = math_ops.to_float(values)
  if weights is not None:
    with ops.control_dependencies((
        check_ops.assert_rank_in(weights, (0, array_ops.rank(values))),)):
      weights = math_ops.to_float(weights)
      values = math_ops.multiply(values, weights)

  value_tensor = array_ops.identity(count)
  update_op = state_ops.assign_add(count, math_ops.reduce_sum(values))

  if metrics_collections:
    ops.add_to_collections(metrics_collections, value_tensor)

  if updates_collections:
    ops.add_to_collections(updates_collections, update_op)

  return value_tensor, update_op

def main(_):
    evaluate(FLAGS.dataset_dir)

if __name__ == '__main__':
    tf.app.run()
