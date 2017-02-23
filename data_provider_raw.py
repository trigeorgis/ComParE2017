from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import wave
import struct
import csv

from pathlib import Path
from menpo.visualize import print_progress
from data_generator import read_wave, get_labels

__signal_framerate = 16000


def get_split(dataset_dir, split_name='train'):
  """ Returns the audio input and labels. 
  
  Args:
      dataset_dir: The directory that contains the data.
      split_name: A train/test/valid split name.
  Returns:
      The raw audio examples and the corresponding labels.
  """

  labels_map = get_labels(Path(dataset_dir) / 'labels.txt')

  files = labels_map[split_name]

  num_classes = len(np.unique([g for _, g in files]))

  for i, (name, label) in enumerate(files):
    path = Path(dataset_dir) / 'wav' / str(name)
    wav = read_wave(path)

    yield (wav, np.eye(num_classes)[label], name)
 
