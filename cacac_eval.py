from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import data_provider
import models
import metrics
import math

from menpo.visualize import print_progress
from pathlib import Path
from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 15, '''The batch size to use.''')
tf.app.flags.DEFINE_string('model', 'audio','''Which model is going to be used: audio,video, or both ''')
tf.app.flags.DEFINE_string('dataset_dir', '/vol/atlas/homes/pt511/db/URTIC/tf_records', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('checkpoint_dir', 'ckpt/conv_x1_reg_1rnn_w/', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('log_dir', 'ckpt/conv_x1_reg_1rnn_w/', 'The tfrecords directory.')
tf.app.flags.DEFINE_integer('num_examples', None, 'The number of examples in the test set')
tf.app.flags.DEFINE_string('eval_interval_secs', 300, 'The number of examples in the test set')
tf.app.flags.DEFINE_string('portion', 'devel', 'The portion of the dataset to use -- `train`, `devel`, or `test`.')
#ckpt/conv_x1_reg_1rnn_w/
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

      pred_argmax = tf.argmax(predictions, axis=1) 
      lab_argmax = tf.argmax(labels, axis=1)

      not_lab_argmax = tf.argmin(labels, axis=1)
      not_pred_argmax = tf.argmin(predictions, axis=1)

      names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
	  'eval/recall1': slim.metrics.streaming_recall(pred_argmax, lab_argmax, name='recall1'),
	  'eval/recall2': slim.metrics.streaming_recall(not_pred_argmax, not_lab_argmax, name='recall2'),
          "eval/accuracy": slim.metrics.streaming_accuracy(pred_argmax, lab_argmax, name='accuracy')
      })

      summary_ops = []
      metrics = dict()
      for name, value in names_to_values.items():
        op = tf.summary.scalar(name, value)
        op = tf.Print(op, [value], name)
        summary_ops.append(op)
        metrics[name] = value

      # Computing the unweighted average recall and add it into summaries
      uar = (metrics['eval/recall1'] + metrics['eval/recall2']) / 2.
      op = tf.summary.scalar('eval/uar', uar)
      op = tf.Print(op, [uar], 'eval/uar')
      summary_ops.append(op)

      num_examples = FLAGS.num_examples or num_examples
      num_batches = math.ceil(num_examples / float(FLAGS.batch_size))
      logging.set_verbosity(1)

#      print(lab_argmax)
#      print(labels)
#      print(predictions)
#      print(pred_argmax)

      ''''
      op = tf.summary.histogram('lab_argmax', lab_argmax)
      op = tf.Print(op, [lab_argmax], 'lab_argmax', summarize=10)
      summary_ops.append(op)

      op = tf.summary.histogram('labels', labels)
      op = tf.Print(op, [labels], 'labels', summarize=10)
      summary_ops.append(op)

      op = tf.summary.histogram('pred_argmax', pred_argmax)
      op = tf.Print(op, [pred_argmax], 'pred_argmax', summarize=10)
      summary_ops.append(op)

      op = tf.summary.histogram('predictions', predictions)
      op = tf.Print(op, [predictions], 'predictions', summarize=10)
      summary_ops.append(op)

      op = tf.summary.histogram('not_pred_argmax', not_pred_argmax)
      op = tf.Print(op, [not_pred_argmax], 'not_pred_argmax', summarize=10)
      summary_ops.append(op)
      '''

      # Setup the global step.
#      slim.get_or_create_global_step()

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
