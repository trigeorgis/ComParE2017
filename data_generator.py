import tensorflow as tf
import numpy as np
import wave

from pathlib import Path

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('wave_folder', 'CACAC/all/', 'The folder that contains the wav files.')
tf.app.flags.DEFINE_string('labels_file', 'CACAC/labels.txt', 'The folder that contains the labels.txt file.')
tf.app.flags.DEFINE_string('tf_folder', 'CACAC/tf_records', 'The folder to write the tf records.')
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
  labels_idx = {'V': 0, 'E': 1, 'O': 2, 'T': 3}

  with open(str(label_file)) as f:
    raw_labels = [np.array(x.strip().split('\t'))[[0, 1]] for x in f.readlines()]

  labels = {}
  for name, g in raw_labels[1:]:
    idx = name.find('_')
    labels.setdefault(name[:idx], []).append((name, labels_idx[str(g)]))

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
   
  if num_of_channels > 1:
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

def serialize_sample(writer, sample_data, root_dir, upsample=False):
  classes = [label for _, label in sample_data]
  class_ids = set(classes)
  num_samples_per_class = {class_name: sum(x == class_name for x in classes) for class_name in class_ids}
  print(num_samples_per_class)
   
  if upsample:
    ratio = list(num_samples_per_class.values())
    ratio = ratio[0] / ratio[1]
    majority_id = ratio > 1

    if ratio < 1:
        ratio  = 1 / ratio
       
    augmented_data = []
   
    for _ in range(int(ratio)):
        for sample, label in sample_data:
            if label == majority_id:
                augmented_data.append((sample, label))
   
    print('Augmented the dataset with {} samples'.format(len(augmented_data)))
    sample_data += augmented_data
   
    import random
    random.shuffle(sample_data)
   
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
  for portion in ['devel', 'train']:
    print('Creating tfrecords for [{}].'.format(portion))
    writer = tf.python_io.TFRecordWriter(
        (Path(tfrecords_folder) / '{}.tfrecords'.format(portion)
    ).as_posix())

    serialize_sample(writer, labels[portion], root_dir, upsample=False)
    writer.close()

if __name__ == '__main__':
  main(FLAGS.wave_folder, FLAGS.labels_file, FLAGS.tf_folder)
