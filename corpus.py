import functools

import tensorflow as tf
from tokenizer_tools.conllz.reader import read_conllz
from tokenizer_tools.converter.conllz_to_offset import conllz_to_offset


def generator_fn(input_file):
    with tf.io.gfile.GFile(input_file) as fd:
        sentence_list = read_conllz(fd)

    for sentence in sentence_list:
        offset_data, result = conllz_to_offset(sentence)

        yield offset_data


class Corpus(object):
    EVAL = 'eval'
    TRAIN = 'train'

    def __init__(self, config):
        self.config = config
        self.dataset_mapping = {}

    def prepare(self):
        self.dataset_mapping[self.TRAIN] = functools.partial(generator_fn, self.config['train'])
        self.dataset_mapping[self.EVAL] = functools.partial(generator_fn, self.config['test'])

    def get_generator_func(self, data_set):
        return self.dataset_mapping[data_set]
