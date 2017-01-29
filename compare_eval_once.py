from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import data_provider
import models
import numpy as np
import sklearn.metrics as sm
import math

from pathlib import Path
from menpo.visualize import print_progress

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 15, '''The batch size to use.''')
tf.app.flags.DEFINE_string('model', 'audio', 'Which model is going to be used: `audio`, `video`, or `both`.')
tf.app.flags.DEFINE_string('dataset_dir', '/vol/atlas/homes/pt511/db/URTIC/tf_records', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('checkpoint_dir', 'ckpt/conv_x1_reg_1rnn_w/', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('portion', 'train', 'The portion of the dataset to use -- `train`, `devel`, or `test`.')

def evaluate(data_folder):
  """Evaluates the model once. Prints in terminal the Accuracy and the UAR of the audio model.
    
  Args:
     data_folder: The folder that contains the test data.
  """

  g = tf.Graph()
  with g.as_default():
    
    # Load dataset.
    audio, labels, num_examples = data_provider.get_split(
        data_folder, FLAGS.portion, FLAGS.batch_size)
    
    # Define model graph.
    with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                           is_training=False):
      predictions = models.get_model(FLAGS.model)(audio)

    coord = tf.train.Coordinator()
    variables_to_restore = slim.get_variables_to_restore()

    num_batches = math.ceil(num_examples / float(FLAGS.batch_size))

    evaluated_predictions = []
    evaluated_labels = []

    saver = tf.train.Saver(variables_to_restore)
    model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    print('Loading model from {}'.format(model_path))
    
    with tf.Session() as sess:
        saver.restore(sess, model_path)  
        tf.train.start_queue_runners(sess=sess)

        try:
            
            for _ in print_progress(range(num_batches), prefix="Batch"):
                pr, l = sess.run([predictions, labels])
                evaluated_predictions.append(pr)
                evaluated_labels.append(l)#

                if coord.should_stop():
                    break
            
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)

        predictions = np.reshape(evaluated_predictions, (-1, 2))
        labels = np.reshape(evaluated_labels, (-1, 2))

        pred_argmax = np.argmax(predictions, axis=1)
        lab_argmax = np.argmax(labels, axis=1)

        not_pred_argmax = np.argmin(predictions, axis=1)
        not_lab_argmax = np.argmin(labels, axis=1)

        correct = (pred_argmax == lab_argmax).mean()
        print('Accuracy: {}'.format(correct))

        recall_1 = sm.recall_score(lab_argmax, pred_argmax)
        recall_2 = sm.recall_score(not_lab_argmax, not_pred_argmax)

        print('Function recall 1: {}'.format(recall_1))
        print('Function recall 2: {}'.format(recall_2))

        uar = (recall_1 + recall_2) / 2
        print('UAR: {}'.format(uar))

def main(_):
    evaluate(FLAGS.dataset_dir)

if __name__ == '__main__':
    tf.app.run()
