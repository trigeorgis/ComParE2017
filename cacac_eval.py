from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import data_provider
import models

from menpo.visualize import print_progress
from pathlib import Path

from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                          '''If specified, restore this pretrained model '''
                          '''before beginning any training.''')
tf.app.flags.DEFINE_integer('batch_size', 15, '''The batch size to use.''')
tf.app.flags.DEFINE_string('model', 'audio','''Which model is going to be used: audio,video, or both ''')
tf.app.flags.DEFINE_string('dataset_dir', 'CACAC/tf_records', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('checkpoint_dir', './ckpt/train/', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('log_dir', './ckpt/logs/valid/', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('num_examples', 3550, 'The number of examples in the test set')
tf.app.flags.DEFINE_string('eval_interval_secs', 300, 'The number of examples in the test set')
tf.app.flags.DEFINE_string('mode', 'devel', 'The number of examples in the test set')

def evaluate(data_folder):

  g = tf.Graph()
  with g.as_default():
    
    # Load dataset.
    audio, labels = data_provider.get_split(data_folder, FLAGS.mode, FLAGS.batch_size)
    
    # Define model graph.
    with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                           is_training=False):
      predictions = models.get_model(FLAGS.model)(audio)

      names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
          "eval/accuracy": slim.metrics.streaming_accuracy(predictions, labels),
         # "eval/TP": slim.metrics.streaming_true_positives(predictions, labels),
         # "eval/FN": slim.metrics.streaming_false_negatives(predictions, labels),
         # "eval/TN": slim.metrics.streaming_true_negatives(predictions, labels),
         # "eval/FP": slim.metrics.streaming_false_positives(predictions, labels),
      })

      # Create the summary ops such that they also print out to std output:
      # Create the summary ops such that they also print out to std output:
      #recall_1 = names_to_values['eval/TP'] / (names_to_values['eval/TP'] + names_to_values['eval/FN'])
      #recall_2 = names_to_values['eval/TN'] / (names_to_values['eval/TN'] + names_to_values['eval/FP'])
      #op2 = tf.summary.scalar('UAR', tf.truediv(tf.add(recall_1, recall_2), 2))

      # Create the summary ops such that they also print out to std output:
      summary_ops = []

      op = tf.summary.scalar(names_to_values.keys()[0], names_to_updates.values()[0])
      op = tf.Print(op, [names_to_updates.values()[0]], names_to_values.keys()[0])
      summary_ops.append(op)
      #summary_ops.append(op2)

      num_examples = FLAGS.num_examples
      num_batches = num_examples / (FLAGS.batch_size)
      logging.set_verbosity(1)
      
      # Setup the global step.
      slim.get_or_create_global_step()
      eval_interval_secs = FLAGS.eval_interval_secs # How often to run the evaluation.
      slim.evaluation.evaluation_loop(
          '',
          FLAGS.checkpoint_dir,
          FLAGS.log_dir,
          num_evals=num_batches,
          eval_op=names_to_updates.values(),
          summary_op=tf.summary.merge(summary_ops),
          eval_interval_secs=eval_interval_secs)



def main(_):
    evaluate(FLAGS.dataset_dir)

if __name__ == '__main__':
    tf.app.run()
