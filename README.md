# Interspeech 2017 - Computational Paralinguistics Challenge (ComParE)

This package provides training and evaluation code for the end-to-end baseline
for the 2017 ComParE challenges.

The 1st challenge comprises recordings of child/adult and adult/adult
conversations – the task is to determine the addressee (child or adult
by an adult).
Recordings of the near field of 61 individuals (babies) were made with
the LENA recording device in real homes. (The number of actual speakers
is unknown.) Overall, ~11,000 (10886) samples (segments) totalling up to
288 minutes are included.

The 2nd challenge comprises recordings of individuals – the task is to 
determine whether the person speaking is cold or not.
The number of actual speakers is 630 (382 males, 248 females), with age 
ranging from 12 to 84 years old. Overall, the corpus consists of ~11,000 
(11283) audio recordings.

The 3rd challenge comprises recordings of snore sounds of individuals by 
their excitation location within the upper airways. The task is to classify 
4 different types of snoring, which are defined based on the VOTE scheme.
There are 843 snore events from 224 subjects.

For questions about these models please contact:
[George Trigeorgis](g.trigeorgis@ic.ac.uk) or
[Panayiotis Tzirakis](panagiotis.tzirakis12@ic.ac.uk)

If you use this codebase in your experiments please cite:

> [**Adieu Features? End-to- End Speech Emotion Recognition using a Deep Convolutional Recurrent Network**
G. Trigeorgis, F. Ringeval, R. B. , E. Marchi, M. Nicoalou a., B. Schuller, S. Zafeiriou. 
*ICASSP. March 2016.*]
(https://ibug.doc.ic.ac.uk/media/uploads/documents/learning_audio_paralinguistics_from_the_raw_waveform.pdf)

1. [Installation](#installation)
2. [Methodology](#methodology)
3. [Running](#running)
4. [Results](#results)
5. [Evaluation](#evaluation)

## 1. Installation
We highly recommended to use [conda](http://conda.pydata.org/miniconda.html) as your Python distribution.
Once downloading and installing [conda](http://conda.pydata.org/miniconda.html), this project can be installed by:

**Step 1:** Create a new conda environment and activate it:
```console
$ conda create -n compare python=3.5
$ source activate compare
```

**Step 2:** Install [TensorFlow](https://www.tensorflow.org/) following the 
official [installation instructions](https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html). 
For example, for 64-bit Linux, the installation of GPU enabled, Python 3.5 TensorFlow involves:
```console
(compare)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.0-cp35-cp35m-linux_x86_64.whl
(compare)$ pip install --upgrade $TF_BINARY_URL
```

**Step 4:** Clone and install the `compare` project as:
```console
(compare)$ git clone git@github.com:trigeorgis/ComParE2017.git
```

## 2. Methodology

We use a convolutional-recurrent architecture which is comprised of convolutional networks
which extract features of the raw waveform, and an LSTM network which takes these features
and classifies the whole sequence as one of the classes in the datasets. 

The waveform is split in 40ms chunks and for each of these we extract features and then 
we use a recurrent network to traverse the whole sequence. At the end we are left with 
the hidden state from the LSTM network which we use to do the final classification.

## 3. Generating data

There are two options to use the input data to run experiments.

The first is to convert the original wave files in a format more suitable for
TensorFlow using TF Records.

> Addressee (First Challenge)
```console
(compare)$ python data_generator.py --wave_folder=path/to/wave_folder --arff_path=ComParE2017_Addressee\* --tf_folder=tf_records 
```

> Cold (Second Challenge)
```console
(compare)$ python data_generator.py --wave_folder=path/to/wave_folder --arff_path=ComParE2017_Cold\* --tf_folder=tf_records 
```
> Snore (Third Challenge)
```console
(compare)$ python data_generator.py --wave_folder=path/to/wave_folder --arff_path=ComParE2017_Snore.ComParE\* --tf_folder=tf_records 
```

By default the `tfrecords` will be generated in a folder called `tf_records` which 
containts a file for each dataset split (`train`, `devel`, `test`).


## 4. Training the models

> Addressee (First Challenge)
```console
(compare)$ python compare_train.py --task=addresee --train_dir=ckpt/train_addresee
```

> Cold (Second Challenge)
```console
(compare)$ python compare_train.py --task=cold --train_dir=ckpt/train_cold
```

> Snore (Third Challenge)
```console
(compare)$ python compare_train.py --task=snore --train_dir=ckpt/train_snore
```

The training script accepts the following list of arguments.

```
  --initial_learning_rate INITIAL_LEARNING_RATE
                        Initial learning rate.
  --batch_size BATCH_SIZE
                        The batch size to use.
  --num_preprocess_threads NUM_PREPROCESS_THREADS
                        How many preprocess threads to use.
  --train_dir TRAIN_DIR
                        Directory where to write event logs and checkpoint.
  --pretrained_model_checkpoint_path PRETRAINED_MODEL_CHECKPOINT_PATH
                        If specified, restore this pretrained model before
                        beginning any training.
  --max_steps MAX_STEPS
                        Number of batches to run.
  --train_device TRAIN_DEVICE
                        Device to train with.
  --model MODEL         Which model is going to be used: audio,video, or both
  --dataset_dir DATASET_DIR
                        The tfrecords directory.
  --task TASK           The task to execute. `addressee`, `cold`, or `snore`.
  --portion PORTION     Dataset portion to use for training (train or devel).
```

## 5. Evaluating the models

While training the models it is useful to run an evaluator service to do continueous 

```console
(compare)$ python compare_eval.py --task=(addresee or cold or snore) --checkpoint_dir=ckpt/train
```

TensorBoard: You can simultaneously run the training and validation. The results can be observed through TensorBoard. Simply run:

```
(compare)$ tensorboard --logdir=ckpt
```

This makes it easy to explore the graph, data, loss evolution and accuracy on the validation set. Once you have a models which performs well on the validation set (which can take between 10k-70k steps depending on the dataset) you can stop the training process.


