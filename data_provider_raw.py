from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import tensorflow as tf
from pathlib import Path
from scipy.io import wavfile
from tensorflow.contrib import ffmpeg

slim = tf.contrib.slim


def read_files(filename_queue):

  audio_binary = tf.read_file(filename_queue[0])
  raw_audio = ffmpeg.decode_audio(audio_binary, file_format='wav', samples_per_second=16000, channel_count=1)
  
  return raw_audio, filename_queue[1] == 'CDS'

def get_split(dataset_dir, split_name='train', batch_size=32):
    """Returns a data split of the RECOLA dataset.
    
    Args:
        split_name: A train/test/valid split name.
    Returns:
        The raw audio examples and the corresponding arousal/valence
        labels.
    """
    with open('CACAC/labels.txt') as f:
      reader = csv.reader(f, delimiter=";")
      files = list(reader)

    files = [(x) for x in files if split_name in x[1]]
    paths = [dataset_dir + str(x[0])+','+str(x[2]) for x in files]
    audio_files = [dataset_dir + str(x[0]) for x in files]
    labels = [str(x[2]) for x in files]

    is_training = split_name == 'train'
    filename_queue = tf.train.slice_input_producer([audio_files, labels],
                                            shuffle=is_training)
    #filename_queue = tf.train.string_input_producer(paths, shuffle=is_training)

    raw_audio, label = read_files(filename_queue)
  
    raw_audio.set_shape([640,1])
    label = tf.cast(label,tf.uint8)
    #label.set_shape([1])    

    raw_audio, label = tf.train.shuffle_batch(
        [raw_audio, label], 1, 5000, 500, 4)

    #raw_audio = tf.decode_raw(raw_audio, tf.float32)

    frames, labels = tf.train.batch([raw_audio, label[0]], batch_size,
                                    capacity=1000, dynamic_pad=True)

    frames = tf.reshape(frames, (batch_size, -1, 640))
    labels = slim.one_hot_encoding(labels, 2)
    
    return frames, labels
