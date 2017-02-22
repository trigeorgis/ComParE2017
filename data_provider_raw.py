from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import wave
import struct
import csv

from pathlib import Path
from menpo.visualize import print_progress

__signal_framerate = 16000

def read_wav_file(wav_file):
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

  fp = wave.open(str(wav_file))
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

def encoder(labels):
  """ Returns one hot encoding of the binary labels.
  
  Args:
      labels: The binary labels.
  Returns:
      One hot encoding of the labels.
  """

  data = []
  for i in range(len(labels)):
    if labels[i]:
      data.append(np.array([1,0]))
    else:
      data.append(np.array([0,1]))
  data = np.vstack(data)
  
  return data

def get_split(dataset_dir, split_name='train', class_name='cold'):
  """ Returns the audio input and labels. 
  
  Args:
      dataset_dir: The directory that contains the data.
      split_name: A train/test/valid split name.
  Returns:
      The raw audio examples and the corresponding labels.
  """

  with open(str(Path(dataset_dir) / 'labels.txt')) as f:
    reader = csv.reader(f, delimiter=";")
    files = list(reader)

  files = [(x) for x in files if split_name in x[1]]

  audio_files = [Path(dataset_dir) / 'wav' / str(x[0]) for x in files]
  labels = [str(x[-1]) == class_name for x in files]
  labels = encoder(labels)

  data = []

  for i, name in enumerate(audio_files):
    name = audio_files[i]
    wav = read_wav_file(name)

    yield (wav, labels[i,:], name)
 
