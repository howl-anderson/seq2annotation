from tokenizer_tools.conllz.reader import read_conllz
import tensorflow as tf


def parse_fn(word_tag_pairs):
    # Encode in Bytes for TF
    words = [i[0] for i in word_tag_pairs]
    tags = [i[1] for i in word_tag_pairs]
    assert len(words) == len(tags), "Words and tags lengths don't match"
    return (words, len(words)), tags


def generator_fn(input_file):
    # with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
    #     for line_words, line_tags in zip(f_words, f_tags):
    #         yield parse_fn(line_words, line_tags)

    with tf.io.gfile.GFile(input_file) as fd:
        sentence_list = read_conllz(fd)

    for sentence in sentence_list:
        # Encode in Bytes for TF
        word_tag_pairs = list(zip(sentence.word_lines, sentence.attribute_lines[0]))

        yield parse_fn(word_tag_pairs)


if __name__ == "__main__":
    # for i in generator_fn('data/corpus/train.txt'):
    #     print(i)

    for i in generator_fn('data/test.conllz'):
        print(i)
