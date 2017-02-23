import tensorflow as tf
import numpy as np
import wave

from pathlib import Path

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('wave_folder', 'wav/', 'The folder that contains the wav files.')
tf.app.flags.DEFINE_string('labels_file', 'labels.txt', 'The path to the labels.txt file.')
tf.app.flags.DEFINE_string('tf_folder', 'tf_records', 'The folder to write the tf records.')

__signal_framerate = 16000

def get_labels(label_path):
  """Parses the data arff files to extract the labels 

  Args:
      label_path: A path glob which contains the arff files with the labels.
  Returns:
      A dictionary for the labels of each fold.
  """
  labels = {}
  class_names = None
  label_path = Path(label_path)
  print('Extracting labels from {}'.format(label_path))

  for path in label_path.parent.glob(label_path.name):
      portion = path.suffixes[-2][1:]
      print('Processing {}'.format(path))
  
      with open(str(path)) as f:
          gts = [np.array(l.strip().split(','))[[0, -1]] for l in f.readlines() if l[0] != '@' and 'wav' in l]
  
      if class_names is None:
          class_names = np.unique([g for _, g in gts])
  
      for name, class_name in gts:
  
          # No labels exist for this dataset.
          if '?' in class_name:
              print('No labels exist for the {} portion'.format(portion))
              break
  
          class_id = np.where(class_name == class_names)[0][0]
          labels.setdefault(portion, []).append((name.replace("'", ""), int(class_id)))

  return labels


def read_wave(path):
  """Reads a wav file and splits it in chunks of 40ms. 
  Pads with zeros if duration does not fit exactly the 40ms chunks.
  Assumptions: 
      A. Wave file has one channel.
      B. Frame rate of wav file is 16KHz.
  
  Args:
      wav_file: The name of the wav file.
  Returns:
      A data array, where each row corresponds to 40ms.
  """

  fp = wave.open(str(path))
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

  return audio.astype(np.float32)

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
    max_samples = np.max(list(num_samples_per_class.values()))
    augmented_data = []

    for class_name, n_samples in num_samples_per_class.items():
        n_samples_to_add = max_samples - n_samples

        while n_samples_to_add > 0:
            for sample, label in sample_data:
                if n_samples_to_add <= 0:
                    break

                if label == class_name:
                    augmented_data.append((sample, label))
                    n_samples_to_add -= 1

    print('Augmented the dataset with {} samples'.format(len(augmented_data)))
    sample_data += augmented_data

    import random
    random.shuffle(sample_data)

  for i, (wav_file, label) in enumerate(sample_data):

    audio = read_wave(root_dir / wav_file)
    example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int_feauture(label),
                'raw_audio': _bytes_feauture(audio.astype(np.float32).tobytes()),
            }))

    writer.write(example.SerializeToString())
    del audio, label

def main(data_folder, labels_file, tfrecords_folder):

  root_dir = Path(data_folder)
  labels = get_labels(labels_file)
  for portion in ['train', 'devel']:
    print('Creating tfrecords for [{}].'.format(portion))
    if not Path(tfrecords_folder).exists():
        Path(tfrecords_folder).mkdir()

    writer = tf.python_io.TFRecordWriter(
        (Path(tfrecords_folder) / '{}.tfrecords'.format(portion)
    ).as_posix())
    
    serialize_sample(writer, labels[portion], root_dir, upsample='train' in portion)
    writer.close()

if __name__ == '__main__':
  main(FLAGS.wave_folder, FLAGS.labels_file, FLAGS.tf_folder)
