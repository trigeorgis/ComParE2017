import tensorflow as tf
import numpy as np
import wave

from pathlib import Path

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('wave_folder', 'CACAC/all/', 'The folder that contains the wav files.')
tf.app.flags.DEFINE_string('labels_file', 'CACAC/labels.txt', 'The folder that contains the labels.txt file.')
tf.app.flags.DEFINE_string('tf_folder', 'CACAC/all/tf_records', 'The folder to write the tf records.')
tf.app.flags.DEFINE_string('class_name', 'CDS', 'The majority class name.')

__signal_framerate = 16000

def get_labels(label_file, class_name):
  """Parses the labels.txt file. 

  Args:
      label_file: The path of the labels.txt file.
      class_name: The name of the minority class.
  Returns:
      A dictionary for the labels of each fold.
  """
  root_dir = Path(root_dir)
  with open((label_file).as_posix()) as f:
    raw_labels = [x.strip().split(';') for x in f.readlines()]
  
  labels = {}
  for name, portion, g in raw_labels:
    labels.setdefault(portion, []).append((name, g == class_name))

  return labels

def get_audio(wav_file, root_dir):
  """Reads a wav file and splits it in chunks of 40ms. 
  Pads with zeros if duration does not fit exactly the 40ms chunks.
  Assumptions: 
      A. Wave file has one channel.
      B. Frame rate of wav file is 16KHz.
  
  Args:
      wav_file: The name of the wav file.
      root_dir: The directory were the wav file is.
  Returns:
      A data array, where each row corresponds to 40ms.
  """

  fp = wave.open(str(root_dir / wav_file))
  num_of_channels = fp.getnchannels()
  fps = fp.getframerate()
    
  if nchan > 1:
    raise ValueError('The wav file should have 1 channel. [{}] found'.format(num_of_channels))

  if fps != __signal_framerate:
    raise ValueError('The wav file should have 16000 fps. [{}] found'.format(fps))

  chunk_size = 640 # 40ms if fps = 16k.

  num_frames = fp.getnframes()
  dstr = fp.readframes(num_frames * num_of_channels)
  data = np.fromstring(dstr, np.int16)
  audio = np.reshape(data, (-1))
  audio = audio / 2.**15 # Normalise audio data (int16).

  audio = np.pad(audio, (0, chunk_size - audio.shape[0] % chunk_size), 'constant')
  audio = audio.reshape(-1, chunk_size)

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
  labels = get_labels(labels_file, FLAGS.class_name)
  for portion in labels:
    print('Creating tfrecords for [{}].'.format(portion))
    writer = tf.python_io.TFRecordWriter(
        (Path(tfrecords_folder) / '{}.tfrecords'.format(portion)
    ).as_posix())
    
    serialize_sample(writer, labels[portion], root_dir)
    writer.close()

if __name__ == '__main__':
  main(FLAGS.wave_folder, FLAGS.labels_folder, FLAGS.tf_folder)
