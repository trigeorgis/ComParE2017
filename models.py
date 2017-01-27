from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

slim = tf.contrib.slim

def recurrent_model(net, hidden_units=64, num_classes=2):
  """Adds the LSTM network on top of the spatial audio model.

  Args:
     net: A `Tensor` of dimensions [batch_size, seq_length, num_features].
     hidden_units: The number of hidden units of the LSTM cell.
     num_classes: The number of classes.
  Returns:
      The prediction of the network.
  """

  batch_size, seq_length, num_features = net.get_shape().as_list()

  lstm = tf.nn.rnn_cell.LSTMCell(hidden_units,
                                 use_peepholes=True,
                                 cell_clip=100,
                                 state_is_tuple=True)

  stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * 2, state_is_tuple=True)


  outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, net, dtype=tf.float32)

  # We have to specify the dimensionality of the Tensor so we can allocate
  # weights for the fully connected layers.
  net = tf.reshape(outputs[:, -1], (batch_size, hidden_units))
  prediction = slim.layers.linear(net, num_classes)

  return tf.reshape(prediction, (batch_size, num_classes))


def audio_model(inputs, conv_filters=32):
    """Creates the audio model.

    Args:
        inputs: A tensor that contains the audio input.
        conv_filters: The number of convolutional filters to use.
    Returns:
        The audio model.
    """

    batch_size, _, num_features = inputs.get_shape().as_list()
    seq_length = tf.shape(inputs)[1]

    net = tf.reshape(inputs, [batch_size * seq_length, 1, num_features, 1])
    net = tf.nn.avg_pool(
        net,
        ksize=[1, 1, 2, 1],
        strides=[1, 1, 2, 1],
        padding='SAME',
        name='subsampling')
    
    with slim.arg_scope([slim.layers.conv2d],
                         padding='SAME', activation_fn=slim.batch_norm):
        for i in range(8):
            net = slim.layers.conv2d(net, conv_filters, (1, 20))

            net = tf.nn.max_pool(
                net,
                ksize=[1, 1, 2, 1],
                strides=[1, 1, 2, 1],
                padding='SAME',
                name='pool')
           
    print(net.get_shape())
    net = tf.reshape(net, (batch_size, seq_length, 2 * 32))
    return net



def get_model(name):
  """ Returns the recurrent audio model.

  Args:
      name: The model to return. Here only 'audio'.
  Returns:
      The recurrent audio model.
  """

  name_to_fun = {'audio': audio_model}

  if name in name_to_fun:
    model = name_to_fun[name]
  else:
    raise ValueError('Requested name [{}] not a valid model'.format(name))

  def wrapper(*args, **kwargs):
    return recurrent_model(model(*args, **kwargs))

  return wrapper

