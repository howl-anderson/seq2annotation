from tokenizer_tools.conll.reader import read_conll
from hanzi_char_lookup_feature import load_trie_from_files, generate_lookup_feature


t = load_trie_from_files({'person': ['/Users/howl/PyCharmProjects/hanzi_char_lookup_feature/data/Chinese_Names_Corpus（120W）.txt']})


def parse_fn(word_tag_pairs):
    # Encode in Bytes for TF
    words = [i[0] for i in word_tag_pairs]
    tags = [i[1] for i in word_tag_pairs]
    assert len(words) == len(tags), "Words and tags lengths don't match"

    words_char = ''.join(words)
    lookup_feature = generate_lookup_feature(t, words_char, ['person'])

    return (words, len(words), lookup_feature['person']), tags


def generator_fn(input_file):
    # with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
    #     for line_words, line_tags in zip(f_words, f_tags):
    #         yield parse_fn(line_words, line_tags)

    sentence_list = read_conll(input_file, sep=None)
    for sentence in sentence_list:
        # Encode in Bytes for TF
        word_tag_pairs = [(i[0], i[1]) for i in sentence]

        yield parse_fn(word_tag_pairs)


if __name__ == "__main__":
    # for i in generator_fn('data/corpus/train.txt'):
    #     print(i)

    for i in generator_fn('data/corpus/test.txt'):
        print(i)
