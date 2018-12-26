from tokenizer_tools.conll.reader import read_conll
from hanzi_char_lookup_feature.n_gram_lookup.load_dicts_from_files import load_dicts_from_files
from hanzi_char_lookup_feature.n_gram_lookup.ngrams_feature import ngrams_feature_mapping
from hanzi_char_lookup_feature.n_gram_lookup.ngrams_feature import generate_lookup_feature, load_data_set


def parse_fn(word_tag_pairs, t, params):
    # Encode in Bytes for TF
    words = [i[0] for i in word_tag_pairs]
    tags = [i[1] for i in word_tag_pairs]
    assert len(words) == len(tags), "Words and tags lengths don't match"

    words_char = ''.join(words)

    lookup_feature = generate_lookup_feature(words_char, 4, t, ['person'], dropout_rate=params['dropout_rate'])

    return (words, len(words), lookup_feature), tags


def generator_fn(input_file, params):
    # with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
    #     for line_words, line_tags in zip(f_words, f_tags):
    #         yield parse_fn(line_words, line_tags)

    t = load_data_set(params['trie_data_mapping'])

    sentence_list = read_conll(input_file, sep=None)
    for sentence in sentence_list:
        # Encode in Bytes for TF
        word_tag_pairs = [(i[0], i[1]) for i in sentence]

        yield parse_fn(word_tag_pairs, t, params)


if __name__ == "__main__":
    # for i in generator_fn('data/corpus/train.txt'):
    #     print(i)

    for i in generator_fn('/Users/howl/PyCharmProjects/seq2annotation/data/test.txt', {'trie_data_mapping': {'person': ['/Users/howl/PyCharmProjects/hanzi_char_lookup_feature/data/THUOCL_lishimingren.txt']}}):
        print(i)
