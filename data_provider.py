from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pathlib import Path

slim = tf.contrib.slim


class Dataset:
    num_classes = 2

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        
    def get_split(self, split_name='train', batch_size=32):
      """Returns a data split of the ComParE dataset.

      Args:
          dataset_dir: The directory that contains the data.
          split_name: One or more train/test/valid split names.
          batch_size: The size of the batch.
      Returns:
          The raw audio examples and the corresponding arousal/valence
          labels.
      """

      paths = [str(Path(self.dataset_dir) / '{}.tfrecords'.format(name)) 
               for name in split_name.split(',')]

      is_training = 'train' in split_name

      filename_queue = tf.train.string_input_producer(paths, shuffle=is_training)

      reader = tf.TFRecordReader()

      _, serialized_example = reader.read(filename_queue)

      features = tf.parse_single_example(
              serialized_example,
              features={
                  'raw_audio': tf.FixedLenFeature([], tf.string),
                  'label': tf.FixedLenFeature([], tf.int64),
                  }
              )

      raw_audio = features['raw_audio']
      label = features['label']

      if is_training:
        raw_audio, label = tf.train.shuffle_batch(
                [raw_audio, label], 1, 1000, 100, 4)
        raw_audio = raw_audio[0]
        label = label[0]

      raw_audio = tf.decode_raw(raw_audio, tf.float32)

      if is_training:
        raw_audio += tf.random_normal(tf.shape(raw_audio), stddev=.25)

      frames, labels = tf.train.batch([raw_audio, label], batch_size,
                                    capacity=1000, dynamic_pad=True)

      frames = tf.reshape(frames, (batch_size, -1, 640))
      labels = slim.one_hot_encoding(labels, self.num_classes)

      return frames, labels, sum(self._split_to_num_samples[name] for name in split_name.split(','))

    
class CACACProvider(Dataset):
    _split_to_num_samples = {
      'test': 3594,
      'devel': 3550,
      'train': 3742
    }

    
class URTICProvider(Dataset):
    _split_to_num_samples = {
      'test': 9551, 
      'devel': 9596, 
      'train': 9505
    }


class SNOREProvider(Dataset):
    num_classes = 4
    _split_to_num_samples = {
      'test': 500, 
      'devel': 644, 
      'train': 500
    }


def get_provider(name):
  """Returns the provider with the given name

  Args:
      name: The provider to return. Here only 'cacac' or 'urtic'.
  Returns:
      The requested provider.
  """

  name_to_class = {'cacac': CACACProvider, 'urtic': URTICProvider,
                   'snore': SNOREProvider}

  if name in name_to_class:
    provider = name_to_class[name]
  else:
    raise ValueError('Requested name [{}] not a valid provider'.format(name))

  return provider

