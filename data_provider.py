from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from pathlib import Path


slim = tf.contrib.slim

_split_to_num_samples = {
    'test': 3594,
    'devel': 3550,
    'train': 3742
}

def get_split(dataset_dir, split_name='train', batch_size=32):
    """Returns a data split of the RECOLA dataset.
    
    Args:
        split_name: A train/test/valid split name.
    Returns:
        The raw audio examples and the corresponding arousal/valence
        labels.
    """
    
    paths = [str(Path(dataset_dir) / '{}.tfrecords'.format(name)) 
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
            [raw_audio, label], 1, 5000, 500, 4)
        raw_audio = raw_audio[0]
        label = label[0]

    raw_audio = tf.decode_raw(raw_audio, tf.float32)

    if is_training:
        raw_audio += tf.random_normal(tf.shape(raw_audio), stddev=.25)

    frames, labels = tf.train.batch([raw_audio, label], batch_size,
                                    capacity=1000, dynamic_pad=True)

    # 640 samples at 16KhZ corresponds to 40ms.
    raw_audio.set_shape([640])
    label.set_shape([])


    frames = tf.reshape(frames, (batch_size, -1, 640))
    labels = slim.one_hot_encoding(labels, 2)
    
    return frames, labels, sum(_split_to_num_samples[name] for name in split_name.split(','))
