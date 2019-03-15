import functools

import tensorflow as tf

from seq2annotation.data_input.char_level_conllz import generator_fn


# def input_fn(input_file, params=None, shuffle_and_repeat=False):
#     params = params if params is not None else {}
#     shapes = (([None], ()), [None])
#     types = ((tf.string, tf.int32), tf.string)
#     defaults = (('<pad>', 0), 'O')
#
#     dataset = tf.data.Dataset.from_generator(
#         functools.partial(generator_fn, input_file),
#         output_shapes=shapes, output_types=types)
#
#     if shuffle_and_repeat:
#         dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])
#
#     dataset = (dataset
#                .padded_batch(params.get('batch_size', 20), shapes, defaults)
#                .prefetch(1))
#
#     feature, label = dataset.make_one_shot_iterator().get_next()
#
#     # vocab_words = tf.contrib.lookup.index_table_from_file(
#     #     params['words'], num_oov_buckets=params['num_oov_buckets'])
#     #
#     # word_ids = vocab_words.lookup(feature[0])
#     #
#     # vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
#     # tags = vocab_tags.lookup(label)
#
#     return {'words': feature[0], 'words_len': feature[1]}, label


def index_table_from_file(vocabulary_file=None):
    index_table = {}
    index_counter = 0
    with open(vocabulary_file) as fd:
        for line in fd:
            key = line.strip()
            index_table[key] = index_counter
            index_counter += 1

    class Lookuper(object):
        def __init__(self, index_table):
            self.index_table = index_table

        def lookup(self, string):
            return self.index_table.get(string)

    return Lookuper(index_table)


def input_fn(params=None, input_file=None, config=None, shuffle_and_repeat=False):
    config = config if config is not None else {}
    shapes = (([None], ()), [None])
    types = ((tf.string, tf.int32), tf.string)
    defaults = (('<pad>', 0), 'O')

    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, input_file),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(config['buffer']).repeat(config['epochs'])

    dataset = (dataset
               .padded_batch(params['batch_size'], shapes, defaults, drop_remainder=config['use_tpu'])
               .prefetch(1))

    words_index_table = index_table_from_file(config['words'])
    tags_index_table = index_table_from_file(config['words'])
    dataset = dataset.map(lambda x, y: ((words_index_table.lookup(x[0]), x[1]), tags_index_table.lookup(y)))

    feature, label = dataset.make_one_shot_iterator().get_next()

    # vocab_words = tf.contrib.lookup.index_table_from_file(
    #     params['words'], num_oov_buckets=params['num_oov_buckets'])
    #
    # word_ids = vocab_words.lookup(feature[0])
    #
    # vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
    # tags = vocab_tags.lookup(label)

    return {'words': feature[0], 'words_len': feature[1]}, label