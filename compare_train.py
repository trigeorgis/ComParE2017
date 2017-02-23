from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import data_provider
import models

from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001, 'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0, 'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_integer('batch_size', 12, '''The batch size to use.''')
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4, 'How many preprocess threads to use.')
tf.app.flags.DEFINE_string('train_dir', 'ckpt/train/',
                           '''Directory where to write event logs '''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           '''If specified, restore this pretrained model '''
                           '''before beginning any training.''')
tf.app.flags.DEFINE_integer('max_steps', 50000, 'Number of batches to run.')
tf.app.flags.DEFINE_string('train_device', '/gpu:0', 'Device to train with.')
tf.app.flags.DEFINE_string('model', 'audio',
                           '''Which model is going to be used: audio,video, or both ''')
tf.app.flags.DEFINE_string('dataset_dir', 'urtic', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('task', 'urtic', 'The task to execute. `cacac`, `urtic`, or `snore`')
tf.app.flags.DEFINE_string('portion', 'train', 'Portion to use for training.')


def train(data_folder):
  """Trains the audio model.

  Args:
     data_folder: The folder that contains the training data.
  """

  g = tf.Graph()
  with g.as_default():
    # Load dataset.
    provider = data_provider.get_provider(FLAGS.task)
    audio, ground_truth, _ = provider(
        data_folder).get_split(FLAGS.portion, FLAGS.batch_size)

    # Define model graph.
    with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                        is_training=True):
        prediction = models.get_model(FLAGS.model)(audio, num_classes=provider.num_classes)

    loss = tf.nn.weighted_cross_entropy_with_logits(prediction, ground_truth,
                                                                pos_weight=1)
    loss = slim.losses.compute_weighted_loss(loss)
    total_loss = slim.losses.get_total_loss()
    
    
    accuracy = tf.reduce_mean(
        tf.to_float(tf.equal(tf.argmax(ground_truth, 1), tf.argmax(prediction, 1))))
    
    chance_accuracy = tf.reduce_mean(
        tf.to_float(tf.equal(tf.argmax(ground_truth, 1), 0)))
    
    tf.scalar_summary('losses/total loss', total_loss)
    tf.scalar_summary('accuracy', accuracy)
    tf.scalar_summary('chance accuracy', chance_accuracy)
    tf.histogram_summary('labels', tf.argmax(ground_truth, 1))
    tf.scalar_summary('losses/Cross Entropy Loss', loss)

    optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

    with tf.Session(graph=g) as sess:
        if FLAGS.pretrained_model_checkpoint_path:
            variables_to_restore = slim.get_variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, FLAGS.pretrained_model_checkpoint_path)

        train_op = slim.learning.create_train_op(total_loss,
                                                 optimizer,
                                                 summarize_gradients=True)

        logging.set_verbosity(1)
        slim.learning.train(train_op,
                            FLAGS.train_dir,
                            save_summaries_secs=60,
                            save_interval_secs=600)

def main(_):
  train(FLAGS.dataset_dir)

if __name__ == '__main__':
  tf.app.run()

