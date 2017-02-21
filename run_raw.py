from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import data_generator as dg
import models
import numpy as np
import os

from pathlib import Path
from menpo.visualize import print_progress

slim = tf.contrib.slim

dataset_dir = Path('/vol/atlas/homes/gt108/db/SNORE/dist/wav/')

def evaluate_raw(file_list):

  g = tf.Graph()
  with g.as_default():

    with slim.arg_scope([slim.batch_norm],
                           is_training=False):
       audio = tf.placeholder(tf.float32, [1, None, 640])
       predictions = models.get_model('audio')(audio)

    coord = tf.train.Coordinator()
    variables_to_restore = slim.get_variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)
    model_path = 'best_models/LSTM2/model.ckpt-28223'
#    model_path = 'best_models/LSTM3/model.ckpt-12024'

    labels_idx = {'0':'V', '1':'E', '2':'O', '3':'T'}
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        tf.train.start_queue_runners(sess=sess)

        try:
          predicts = []
          for i in print_progress(range(len(file_list))):
            wav_file = Path(file_list[i])
            input = dg.get_audio(wav_file, dataset_dir)
            input = np.expand_dims(input, axis=0)

            pr = sess.run([predictions], feed_dict={audio:input})
            a = np.argmax(pr)

            predicts.append(labels_idx[str(a)])

            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)

    return predicts

def main():
  file_list = [x for x in os.listdir(str(dataset_dir)) if 'devel' in x]

#  labels_idx = {'V': 0, 'E': 1, 'O': 2, 'T': 3}
#  labels_idx = {'0':'V', '1':'E', '2':'O', '3':'T'}
  pr = evaluate_raw(file_list)
  print(pr)

  files = np.array(file_list).reshape((-1,1))
  predictions = np.array(pr).reshape((-1,1))

  data = np.hstack([files, predictions])
  print(data)
#  np.savetxt('LSTM2_results_devel.txt', data,  fmt='%.18s')

if __name__ == '__main__':
  main()
