import collections
import functools
import json
import logging
from collections import OrderedDict
from typing import Dict, List

import tensorflow as tf

from seq2annotation.utils import class_from_module_path, load_hook
from tokenizer_tools.tagset.converter.offset_to_biluo import offset_to_biluo
from tokenizer_tools.tagset.NER.BILUO import BILUOEncoderDecoder
from nlp_utils.preprocess.lookup_table import LookupTable as Lookuper

logger = logging.getLogger(__name__)


def index_table_from_file(vocabulary_file=None, config={}):
    index_table = {}
    index_counter = 0
    with open(vocabulary_file) as fd:
        for line in fd:
            key = line.strip('\n')
            index_table[key] = index_counter
            index_counter += 1

    return Lookuper(index_table, **config)


def read_assets():
    return {
        'vocab_filename': 'data/unicode_char_list.txt',
        'tag_filename': 'data/tags.txt'
    }


def generator_func(data_generator_func, config):
    # load plugin
    preprocess_hook = load_hook(config.get('preprocess_hook', []))

    for sentence in data_generator_func():
        for hook in preprocess_hook:
            sentence = hook(sentence)

        if isinstance(sentence, list):
            for s in sentence:
                yield parse_fn(s)
        else:
            yield parse_fn(sentence)


def parse_fn(offset_data):
    tags = offset_to_biluo(offset_data)
    words = offset_data.text
    assert len(words) == len(tags), "Words and tags lengths don't match"

    logger.debug((words, len(words)), tags)

    return (words, len(words)), tags


def parse_to_dataset(data_generator_func, config=None, shuffle_and_repeat=False):
    config = config if config is not None else {}
    shapes = (([None], ()), [None])
    types = ((tf.string, tf.int32), tf.string)
    defaults = (('<pad>', 0), 'O')

    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_func, data_generator_func, config),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        # print(">>> {}".format(config))
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

    # word_index_lookuper = tf.contrib.lookup.index_table_from_file(
    #     read_assets()['vocab_filename'],
    #     num_oov_buckets=1
    # )
    # words = word_index_lookuper.lookup(words)
    #
    # tag_index_lookuper = tf.contrib.lookup.index_table_from_file(
    #     read_assets()['tag_filename'],
    #     num_oov_buckets=1
    # )
    # label = tag_index_lookuper.lookup(label)

    return {'words': words, 'words_len': words_len}, label


def build_input_func(data_generator_func, config=None):
    def input_func():
        train_dataset = parse_to_dataset(data_generator_func, config, shuffle_and_repeat=True)
        data_iterator = dataset_to_feature_column(train_dataset)

        return data_iterator

    return input_func


def build_gold_generator_func(offset_dataset):
    return functools.partial(generator_func, offset_dataset)


def generate_tagset(tags) -> List[str]:
    if not tags:
        # empty entity still have O tag
        return [BILUOEncoderDecoder.oscar]

    tagset = set()
    for tag in tags:
        encoder = BILUOEncoderDecoder(tag)
        tagset.update(encoder.all_tag_set())

    tagset_list = list(tagset)

    # make sure O is first tag,
    # this is a bug feature, otherwise sentence_correct is not correct
    # due to the crf decoder, need fix
    tagset_list.remove(BILUOEncoderDecoder.oscar)
    tagset_list = list(sorted(tagset_list, key=lambda x: x))

    tagset_list.insert(0, BILUOEncoderDecoder.oscar)

    return tagset_list
