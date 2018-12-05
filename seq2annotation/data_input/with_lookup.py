import functools

import tensorflow as tf

from seq2annotation.data_input.char_and_lookup import generator_fn


def input_fn(input_file, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = (([None], (), [None]), [None])
    types = ((tf.string, tf.int32, tf.string), tf.string)
    defaults = (('<pad>', 0, 'O'), 'O')

    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, input_file, params),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))

    feature, label = dataset.make_one_shot_iterator().get_next()

    # vocab_words = tf.contrib.lookup.index_table_from_file(
    #     params['words'], num_oov_buckets=params['num_oov_buckets'])
    #
    # word_ids = vocab_words.lookup(feature[0])
    #
    # vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
    # tags = vocab_tags.lookup(label)

    return {'words': feature[0], 'words_len': feature[1], 'lookup': feature[2]}, label