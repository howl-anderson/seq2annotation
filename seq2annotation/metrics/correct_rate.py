import numpy as np

import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError
from tensorflow.python.training.monitored_session import MonitoredSession

from tokenizer_tools.hooks import TensorObserveHook


def correct_rate(labels, predictions, weights):
    weights = tf.ones_like(weights, tf.int32)

    labels = tf.cast(labels, tf.int32)
    # with tf.Session() as sess:
    #     sess.run(tf.tables_initializer())
    #     result = sess.run((labels, weights))
    #     print(result)

    label_seq = tf.multiply(labels, weights, name='labels')

    predictions = tf.cast(predictions, tf.int32)
    prediction_seq = tf.multiply(predictions, weights, name='predictions')

    flag = tf.math.equal(label_seq, prediction_seq)

    correct = tf.math.reduce_all(flag, -1)
    
    fake_predictions = tf.to_int32(correct, name='fake_prediction')

    fake_golden = tf.ones(tf.shape(fake_predictions), dtype=tf.int32, name='fake_golden')

    # print(fake_golden, fake_predictions)

    # return tf.metrics.accuracy(fake_golden, fake_predictions), fake_golden, fake_predictions
    return tf.metrics.accuracy(fake_golden, fake_predictions)


if __name__ == "__main__":
    def generator_fn():
        yield ([[1, 2, 3], [1, 1, 0], [1, 1, 1]],
               [[1, 2, 1], [1, 1, 0], [1, 1, 1]])
        yield ([[0, 1, 0], [1, 0, 2], [3, 0, 0]],
               [[0, 1, 0], [2, 1, 2], [0, 3, 1]])

    ds = tf.data.Dataset.from_generator(generator_fn, (tf.int32, tf.int32), ([3, 3], [3, 3]))
    y_true, y_pred = ds.make_one_shot_iterator().get_next()

    # result, fake_labels, fake_predictions = correct_rate(y_true, y_pred)
    result = correct_rate(y_true, y_pred)

    with MonitoredSession(hooks=[TensorObserveHook()]) as sess:
        while not sess.should_stop():
            try:
                sess.run([result[1]])
            except OutOfRangeError as e:
                break

        result = sess.run(result[0])
        # Check final value
        assert np.allclose(result, 0.5)

    # with tf.Session() as sess:
    #     # Initialize and run the update op on each batch
    #     sess.run(tf.local_variables_initializer())
    #     while True:
    #         try:
    #             sess.run([result[1]])
    #         except OutOfRangeError as e:
    #             break
    #
    #     result = sess.run(result[0])
    #     # Check final value
    #     assert np.allclose(result, 0.5)
