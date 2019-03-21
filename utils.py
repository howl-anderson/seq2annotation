import functools

import tensorflow as tf
import tensorflow_datasets as tfds
from tokenizer_tools.tagset.converter.offset_to_biluo import offset_to_biluo


def read_configure():
    return {
        'corpus': {
            'train': './data/train.conllz',
            'test': './data/test.conllz'
        },
        'model': {
            'shuffle_pool_size': 10,
            'batch_size': 32,
            'epochs': 1,
            'arch': {}
         }
    }


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


def read_assets():
    return {
        'vocab_filename': 'data/unicode_char_list.txt',
        'tag_filename': 'data/tags.txt'
    }


def generator_func(data_generator_func):
    for sentence in data_generator_func():
        yield parse_fn(sentence)


def parse_fn(offset_data):
    tags = offset_to_biluo(offset_data)
    words = offset_data.text
    assert len(words) == len(tags), "Words and tags lengths don't match"
    return (words, len(words)), tags


def parse_to_dataset(data_generator_func, config=None, shuffle_and_repeat=False):
    config = config if config is not None else {}
    shapes = (([None], ()), [None])
    types = ((tf.string, tf.int32), tf.string)
    defaults = (('<pad>', 0), 'O')

    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_func, data_generator_func),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        print(">>> {}".format(config))
        dataset = dataset.shuffle(config['shuffle_pool_size']).repeat(config['epochs'])

    # char_encoder = tfds.features.text.SubwordTextEncoder.load_from_file(read_assets()['vocab_filename'])
    # tag_encoder = tfds.features.text.SubwordTextEncoder.load_from_file(read_assets()['tag_filename'])
    # dataset = dataset.map(lambda x: (char_encoder.encode(x[0][0]), tag_encoder.encode(x[0][1]), x[1]))

    # words_index_table = index_table_from_file(read_assets()['vocab_filename'])
    # tags_index_table = index_table_from_file(read_assets()['tag_filename'])
    # dataset = dataset.map(lambda x, y: ((words_index_table.lookup(x[0]), x[1]), tags_index_table.lookup(y)))

    dataset = (dataset
               .padded_batch(config['batch_size'], shapes, defaults)
               .prefetch(1))

    return dataset


def dataset_to_feature_column(dataset):
    (words, words_len), label = dataset.make_one_shot_iterator().get_next()

    word_index_lookuper = tf.contrib.lookup.index_table_from_file(
        read_assets()['vocab_filename'],
        num_oov_buckets=1
    )
    words = word_index_lookuper.lookup(words)

    tag_index_lookuper = tf.contrib.lookup.index_table_from_file(
        read_assets()['tag_filename'],
        num_oov_buckets=1
    )
    label = tag_index_lookuper.lookup(label)

    return {'words': words, 'words_len': words_len}, label


def build_input_func(data_generator_func, config=None):
    def input_func():
        train_dataset = parse_to_dataset(data_generator_func, config, shuffle_and_repeat=True)
        data_iterator = dataset_to_feature_column(train_dataset)
        
        return data_iterator
    
    return input_func


def build_gold_generator_func(offset_dataset):
    return functools.partial(generator_func, offset_dataset)
