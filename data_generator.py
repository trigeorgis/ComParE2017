import tensorflow as tf
import numpy as np
import argparse

from moviepy.editor import AudioFileClip
from pathlib import Path

# Create FLAGS
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('db_folder', 'CACAC/all/', 'The folder that contains the wav files.')
tf.app.flags.DEFINE_string('labels_folder', 'CACAC/', 'The folder that contains the labels.txt file.')
tf.app.flags.DEFINE_string('tf_folder', 'CACAC/all/tf_records', 'The folder to write the tf records.')

def get_labels(root_dir):
  root_dir = Path(root_dir)
  with open((root_dir / 'labels.txt').as_posix()) as f:
    raw_labels = [x.strip().split(';') for x in f.readlines()]
    
    labels = {}
    for name, portion, g in raw_labels:
      labels.setdefault(portion, []).append((name, g == 'CDS'))

  return labels

def get_audio(wav_file, root_dir):
  file = AudioFileClip(str(root_dir /  wav_file))
  file = file.set_fps(16000)
  audio = np.array(list(file.iter_frames()))[:, 0]
  audio = np.pad(audio, (0, 640 - audio.shape[0] % 640), 'constant')
  audio = audio.reshape(-1, 640)

  return audio

def _int_feauture(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feauture(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_sample(writer, sample_data, root_dir):
  for i, (wav_file, label) in enumerate(sample_data):

    audio = get_audio(wav_file, root_dir)
    example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int_feauture(label),
                'raw_audio': _bytes_feauture(audio.tobytes()),
            }))

    writer.write(example.SerializeToString())
    del audio, label

def main(data_folder, labels_file, tfrecords_folder):

  root_dir = Path(data_folder)
  labels = get_labels(labels_file)
  for portion in labels:
    print('Createing {} tfrecord file'.format(portion))
    writer = tf.python_io.TFRecordWriter(
        (Path(tfrecords_folder) / '{}.tfrecords'.format(portion)
    ).as_posix())
    
    serialize_sample(writer, labels[portion], root_dir)
    writer.close()

if __name__ == '__main__':
  main(FLAGS.db_folder, FLAGS.labels_folder, FLAGS.tf_folder)
