from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import data_provider
import models
import numpy as np
import sklearn.metrics as sm

#from menpo.visualize import print_progress
from pathlib import Path

from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                          '''If specified, restore this pretrained model '''
                          '''before beginning any training.''')
tf.app.flags.DEFINE_integer('batch_size', 30, '''The batch size to use.''')
tf.app.flags.DEFINE_string('model', 'audio','''Which model is going to be used: audio,video, or both ''')
tf.app.flags.DEFINE_string('dataset_dir', '/vol/atlas/homes/gt108/db/CACAC/tf_records', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('checkpoint_dir', './ckpt/cov_filt_80/', 'The tfrecords directory.')
tf.app.flags.DEFINE_string('log_dir', './ckpt/cov_filt_80/valid/', 'The tfrecords directory.')
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

      logging.set_verbosity(1)
      
    with tf.Session() as sess:
      coord = tf.train.Coordinator()
      variables_to_restore = slim.get_variables_to_restore()

      saver = tf.train.Saver(variables_to_restore)
      model_path = slim.evaluation.tf_saver.get_checkpoint_state(FLAGS.checkpoint_dir).model_checkpoint_path
      m_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
      print('model_path : {}'.format(model_path))
      print('m_path : {}'.format(m_path))
      saver.restore(sess, model_path)        
      _ = tf.train.start_queue_runners(sess=sess)

      try:
        num_examples = FLAGS.num_examples
        num_batches = int(np.ceil(num_examples / (FLAGS.batch_size)))
        pred = []
        lab = []
        step = 0
        while step < num_batches and not coord.should_stop():
          print('Batch number : {}/{}'.format(step,num_batches))
          pr, l = sess.run([predictions, labels])
          pred.append(pr)
          lab.append(l)
          step += 1

        coord.request_stop()
      except Exception as e:
        coord.request_stop(e)

      predictions_flattened = np.reshape(pred,[-1,2])
      labels_flattened = np.reshape(lab,[-1,2])
      #print(labels_flattened)
      #print(predictions_flattened)
      pred_argmax = np.argmax(predictions_flattened,1)
      lab_argmax = np.argmax(labels_flattened,1)
      
      #print(lab_argmax)
      #print(pred_argmax)

      correct = sum(np.equal(pred_argmax, lab_argmax))/float(FLAGS.num_examples)
      print('Accuracy : {}'.format(correct))

      recall_1 = sm.recall_score(lab_argmax, pred_argmax)
      recall_2 = sm.recall_score(lab_argmax, pred_argmax, pos_label = 0)
      #recall_3 = sm.recall_score(pred_argmax, lab_argmax)
      #print('RECALL 1: {}'.format(recall_1))
      #print('RECALL 2: {}'.format(recall_2))
      #print('RECALL 3: {}'.format(recall_3))

      uar = (recall_1+recall_2)/2
      print('UAR: {}'.format(uar))

        





def main(_):
    evaluate(FLAGS.dataset_dir)

if __name__ == '__main__':
    tf.app.run()
