from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import wave
import struct


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
