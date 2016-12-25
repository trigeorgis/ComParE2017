from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import wave
import struct
import csv
from pathlib import Path


def read_wav_file(wav_file):
    """Read a wav file.
    
    Args:
        wav_file: The name of the wav file
    Returns:
        The audio data.
    """

    fp = wave.open(wav_file)
    nchan = fp.getnchannels()
    N = fp.getnframes()
    dstr = fp.readframes(N*nchan)
    data = np.fromstring(dstr, np.int16)
    audio = np.reshape(data, (-1))

    audio = np.pad(audio, (0, 640 - audio.shape[0] % 640), 'constant')
    audio = audio.reshape(-1, 640)

    return audio

def encoder(labels):

  data = []
  for i in range(len(labels)):
    if labels[i] == True:
      data.append(np.array([1,0]))
    else:
      data.append(np.array([0,1]))
  data = np.vstack(data)
  
  return data

def get_split(dataset_dir, split_name='train'):

  with open('CACAC/labels.txt') as f:
    reader = csv.reader(f, delimiter=";")
    files = list(reader)

  files = [(x) for x in files if split_name in x[1]]
  audio_files = [dataset_dir + split_name +'/'+ str(x[0]) for x in files]
  labels = [str(x[2]) =='CDS' for x in files if str(x[1]) in split_name]
  labels = encoder(labels)

  is_training = split_name == 'train'
  data = []
  for i in range(len(audio_files)):
    wav = read_wav_file(audio_files[i])
    data.append(wav)
  
  data = np.vstack(data)
  
  return data, labels
