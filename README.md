# CHILD OR ADULT CONVERSATIONAL ADDRESSEE CORPUS (CACAC) Challenge
This package provides training and evaluation code for the end-to-end baseline
for the 1st ComParE challenge. 

This challenge comprises recordings of child/adult and adult/adult
conversations – the task is to determine the addressee (child or adult
by an adult).
Recordings of the near field of 61 individuals (babies) were made with
the LENA recording device in real homes. (The number of actual speakers
is unknown.) Overall, ~11,000 (10886) samples (segments) totalling up to
288 minutes are included.


1. [Installation](#installation)
2. [Methodology](#methodology)
3. [Running](#running)
4. [Results](#results)

## 1. Installation
We highly recommended to use [conda](http://conda.pydata.org/miniconda.html) as your Python distribution.
Once downloading and installing [conda](http://conda.pydata.org/miniconda.html), this project can be installed by:

**Step 1:** Create a new conda environment and activate it:
```console
$ conda create -n cacac python=3.5
$ source activate cacac
```

**Step 2:** Install [TensorFlow](https://www.tensorflow.org/) following the 
official [installation instructions](https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html). 
For example, for 64-bit Linux, the installation of GPU enabled, Python 3.5 TensorFlow involves:
```console
(cacac)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.0-cp35-cp35m-linux_x86_64.whl
(cacac)$ pip install --upgrade $TF_BINARY_URL
```

**Step 4:** Clone and install the `cacac` project as:
```console
(cacac)$ git clone git@github.com:trigeorgis/Interspeech2017.git
```

## 2. Methodology

We use a convolutional-recurrent architecture which is comprised of a convolutional networks
which extracts features of the raw waveform and an LSTM network which takes these features
and classifies the whole sequence as one of the two classes (child/adult or adult/adult).

The waveform is split in 40ms chunks and for each of these we extract features and then 
we use a recurrent network to traverse the whole sequence. At the end we are left with 
the hidden state from the LSTM network which we use to do the final classification.

## 3. Running

First we need to convert the original wave files in a format more suitable for
TensorFlow using TF Records.

```console
(cacac)$ python data_generator.py --wave_folder=wav_files --label_file=labels.txt
```

By default the `tfrecords` will be generated in a folder called `tf_records` which 
containts a file for each dataset split (`train`, `devel`, `test`).

## 4. Results
