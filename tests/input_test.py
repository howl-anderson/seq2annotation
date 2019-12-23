from tokenizer_tools.tagset.offset.sequence import Sequence
from seq2annotation.input import generator_func, build_input_func
from tokenizer_tools.tagset.offset.span import Span
import tensorflow as tf


def test_generator_func():
    def data_generator_func():
        seq = Sequence("王小明在北京的清华大学读书。")
        seq.span_set.append(Span(0, 3, 'PERSON'))
        seq.span_set.append(Span(4, 6, 'GPE'))
        seq.span_set.append(Span(7, 11, 'ORG'))

        return [seq]

    config = {
        'preprocess_hook': [{
            'class':
            'seq2annotation.preprocess_hooks.corpus_augment.CorpusAugment'
        }]
    }

    result = generator_func(data_generator_func, config)

    result = [i for i in result]

    assert len(result) == 7


def test_build_input_func():
    def data_generator_func():
        seq = Sequence("王小明在北京的清华大学读书。")
        seq.span_set.append(Span(0, 3, 'PERSON'))
        seq.span_set.append(Span(4, 6, 'GPE'))
        seq.span_set.append(Span(7, 11, 'ORG'))

        return [seq]

    output_func = build_input_func(data_generator_func, {"shuffle_pool_size": 1, "epochs": 2, "batch_size": 1})
    output = output_func()

    with tf.Session() as sess:
        word_info, encoding = sess.run(output)
        assert word_info["words"].shape == (1, 14)
        assert word_info["words_len"].shape == (1,)
        assert encoding.shape == (1, 14)
