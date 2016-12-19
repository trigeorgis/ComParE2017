from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v1

slim = tf.contrib.slim


def recurrent_model(net, hidden_units=256, num_classes=2):
    """Complete me...

    Args:
    Returns:
    """
    batch_size, seq_length, num_features = net.get_shape().as_list()

    lstm = tf.nn.rnn_cell.LSTMCell(hidden_units,
                                   use_peepholes=True,
                                   cell_clip=100,
                                   state_is_tuple=True)

    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * 2, state_is_tuple=True)

    # We have to specify the dimensionality of the Tensor so we can allocate
    # weights for the fully connected layers.
    outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, net, dtype=tf.float32)

    net = tf.reshape(outputs[:, -1], (batch_size, hidden_units))
    net = slim.dropout(net)

    prediction = slim.layers.linear(net, num_classes)
    return tf.reshape(prediction, (batch_size, num_classes))


def audio_model(inputs, conv_filters=40):
    """Complete me...

    Args:
    Returns:
    """

    batch_size, _, num_features = inputs.get_shape().as_list()
    seq_length = tf.shape(inputs)[1]

    audio_input = tf.reshape(inputs, [batch_size * seq_length, 1, num_features, 1])

    with slim.arg_scope([slim.layers.conv2d], padding='SAME'):
        net = slim.batch_norm(audio_input)
        net = slim.layers.conv2d(net, conv_filters, (1, 20))
        net = slim.batch_norm(net)

        # Subsampling of the signal to 8KhZ.
        net = tf.nn.max_pool(
            net,
            ksize=[1, 1, 2, 1],
            strides=[1, 1, 2, 1],
            padding='SAME',
            name='pool1')

        # Original model had 400 output filters for the second conv layer
        # but this trains much faster and achieves comparable accuracy.
        net = slim.layers.conv2d(net, conv_filters, (1, 40))

        net = tf.reshape(net, (batch_size * seq_length, num_features // 2, conv_filters, 1))

        # Pooling over the feature maps.
        net = tf.nn.max_pool(
            net,
            ksize=[1, 1, 10, 1],
            strides=[1, 1, 10, 1],
            padding='SAME',
            name='pool2')

    net = tf.reshape(net, (batch_size, seq_length, num_features // 2 * 4))

    return net


def get_model(name):
    """Complete me...

    Args:
    Returns:
    """
    name_to_fun = {'audio': audio_model}

    if name in name_to_fun:
        model = name_to_fun[name]
    else:
        raise ValueError('Requested name [{}] not a valid model'.format(name))

    def wrapper(*args, **kwargs):
        return recurrent_model(model(*args, **kwargs))

    return wrapper

