import functools

import tensorflow as tf

relu6 = functools.partial(tf.keras.activations.relu, max_value=6)

# set __name__ which will used by tensorflow.python.keras.activations.serialize
relu6.__name__ = "seq2annotation.tf.python.keras.activations.relu6"

tf.keras.utils.get_custom_objects()[relu6.__name__] = relu6
