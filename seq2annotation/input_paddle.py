import functools

from seq2annotation.utils import load_hook
from tokenizer_tools.tagset.converter.offset_to_biluo import offset_to_biluo
from tokenizer_tools.tagset.NER.BILUO import BILUOEncoderDecoder


def generator_func(data_generator_func, config, vocabulary_lookup, tag_lookup):
    # load plugin
    preprocess_hook = load_hook(config.get('preprocess_hook', []))

    for sentence in data_generator_func():
        for hook in preprocess_hook:
            sentence = hook(sentence)

        if isinstance(sentence, list):
            for s in sentence:
                yield parse_fn(s, vocabulary_lookup, tag_lookup)
        else:
            yield parse_fn(sentence, vocabulary_lookup, tag_lookup)


def parse_fn(offset_data, vocabulary_lookup, tag_lookup):
    tags = offset_to_biluo(offset_data)
    words = offset_data.text
    assert len(words) == len(tags), "Words and tags lengths don't match"

    words_id = [vocabulary_lookup.lookup(i) for i in words]
    tags_id = [tag_lookup.lookup(i) for i in tags]

    return words_id, tags_id


class Vocabulary(object):
    def __init__(self, lookup_table):
        self.lookup_table = lookup_table
        self.reverse_lookup_table = {v: k for k, v in lookup_table.items()}

    def lookup(self, str_):
        if str_ in self.lookup_table:
            return self.lookup_table[str_]
        else:
            # not in the table: return extra max(id) + 1
            return len(self.lookup_table)

    def length(self):
        return len(self.lookup_table)

    def id_to_str(self, id_):
        if id_ in self.reverse_lookup_table:
            return self.reverse_lookup_table[id_]
        else:
            return '<UNK>'


def read_vocabulary(vocabulary):
    data = {}

    fd = open(vocabulary) if isinstance(vocabulary, str) else vocabulary
    for line in fd:
        word = line.strip()
        data[word] = len(data)

    if isinstance(vocabulary, str):
        fd.close()

    return Vocabulary(data)


def build_input_func(data_generator_func, config=None):
    vocabulary_lookup = read_vocabulary(config['vocab_data'])
    tag_lookup = read_vocabulary(config['tags_data'])

    def input_func():
        train_dataset = generator_func(data_generator_func, config, vocabulary_lookup, tag_lookup)

        return train_dataset
    
    return input_func


def build_gold_generator_func(offset_dataset):
    return functools.partial(generator_func, offset_dataset)


def generate_tagset(tags):
    if not tags:
        return []

    tagset = set()
    for tag in tags:
        encoder = BILUOEncoderDecoder(tag)
        tagset.update(encoder.all_tag_set())

    tagset_list = list(tagset)
    
    # make sure O is first tag,
    # this is a bug feature, otherwise sentence_correct is not correct
    # due to the crf decoder, need fix
    tagset_list.remove(BILUOEncoderDecoder.oscar)
    tagset_list.insert(0, BILUOEncoderDecoder.oscar)

    return tagset_list

