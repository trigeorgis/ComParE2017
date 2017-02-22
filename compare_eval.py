from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import data_provider
import models
import math

from menpo.visualize import print_progress
from pathlib import Path
from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 1, '''The batch size to use.''')
tf.app.flags.DEFINE_string('model', 'audio','''Which model is going to be used: audio, video, or both ''')
tf.app.flags.DEFINE_string('dataset_dir', '/vol/atlas/homes/pt511/db/URTIC/tf_records', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('checkpoint_dir', 'ckpt/train/', 'The checkpoint directory.')
tf.app.flags.DEFINE_string('log_dir', 'ckpt/eval/', 'The directory to save the event files.')
tf.app.flags.DEFINE_integer('num_examples', None, 'The number of examples in the given portion.')
tf.app.flags.DEFINE_string('eval_interval_secs', 300, 'The number of examples in the test set')
tf.app.flags.DEFINE_string('portion', 'devel', 'The portion of the dataset to use -- `train`, `devel`, or `test`.')
tf.app.flags.DEFINE_string('task', 'urtic', 'The task to execute. `cacac` or `urtic`')

def evaluate(data_folder):
  """Evaluates the audio model.

  Args:
     data_folder: The folder that contains the data to evaluate the audio model.
  """

  g = tf.Graph()
  with g.as_default():
    # Load dataset.
    provider = data_provider.get_provider(FLAGS.task)(data_folder)
    num_classes = provider.num_classes
    audio, labels, num_examples = provider.get_split(FLAGS.portion, FLAGS.batch_size)

    # Define model graph.
    with slim.arg_scope([slim.batch_norm],
                           is_training=False):
      predictions = models.get_model(FLAGS.model)(audio, num_classes=provider.num_classes)
      pred_argmax = tf.argmax(predictions, 1) 
      lab_argmax = tf.argmax(labels, 1)

      metrics = {
          "eval/accuracy": slim.metrics.streaming_accuracy(pred_argmax, lab_argmax, name='accuracy')
      }

      for i in range(num_classes):
          name ='eval/recall_{}'.format(i)
          recall = slim.metrics.streaming_recall(
                  tf.to_int64(tf.equal(pred_argmax, i)),
                  tf.to_int64(tf.equal(lab_argmax, i)), name=name)
          metrics[name] = recall

      names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(metrics)

      summary_ops = []
      metrics = dict()
      for name, value in names_to_values.items():
        op = tf.summary.scalar(name, value)
        op = tf.Print(op, [value], name)
        summary_ops.append(op)
        metrics[name] = value

      # Computing the unweighted average recall and add it into the summaries.
      uar = sum([metrics['eval/recall_{}'.format(i)] for i in range(num_classes)]) / num_classes
      op = tf.summary.scalar('eval/uar', uar)
      op = tf.Print(op, [uar], 'eval/uar')
      summary_ops.append(op)

      num_examples = FLAGS.num_examples or num_examples
      num_batches = math.ceil(num_examples / float(FLAGS.batch_size))
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

def main(_):
    evaluate(FLAGS.dataset_dir)

if __name__ == '__main__':
    tf.app.run()
