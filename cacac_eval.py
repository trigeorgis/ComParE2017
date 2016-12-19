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
tf.app.flags.DEFINE_string('dataset_dir', './tf_records/', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('checkpoint_dir', './ckpt/train/', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('log_dir', './ckpt/logs/valid/', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('num_examples', 10000, 'The number of examples in the test set')
tf.app.flags.DEFINE_string('eval_interval_secs', 300, 'The number of examples in the test set')

def evaluate(data_folder):

  g = tf.Graph()
  with g.as_default():
    
    # Load dataset.
    audio, ground_truth = data_provider.get_split(data_folder, 'train', FLAGS.batch_size)
    
    # Define model graph.
    with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                           is_training=False):
      prediction = models.get_model(FLAGS.model)(audio)

      '''
      num_examples = FLAGS.num_examples
      num_batches = num_examples / (FLAGS.batch_size)
      logging.set_verbosity(1)
      
      
      # Evaluate model
      correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(ground_truth, 1))
      
      for i in range(num_batches):
        acc = tf.add(acc, sess.run(tf.cast(correct_pred, tf.float32)))

      accuracy = tf.reduce_mean(acc)        
      '''


      names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({"eval/precision": slim.metrics.streaming_precision(prediction, ground_truth)})
      
      # Create the summary ops such that they also print out to std output:
      summary_ops = []

      op = tf.summary.scalar(names_to_values.keys()[0], names_to_updates.values()[0])
      op = tf.Print(op, [names_to_updates.values()[0]], names_to_values.keys()[0])
      summary_ops.append(op)

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
