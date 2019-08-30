import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, Bidirectional, LSTM, Input
from tensorflow.python.keras.models import Sequential
from ioflow.configure import read_configure
from ioflow.corpus import get_corpus_processor
from seq2annotation.input import generate_tagset, Lookuper, \
    index_table_from_file
from tf_crf_layer.layer import CRF
from tf_crf_layer.loss import crf_loss
from tf_crf_layer.metrics import crf_accuracy
from tokenizer_tools.tagset.converter.offset_to_biluo import offset_to_biluo
from tf_crf_layer.crf_helper import allowed_transitions, constraint_type

from seq2annotation.reportor import classification_report

config = read_configure()

corpus = get_corpus_processor(config)
corpus.prepare()
train_data_generator_func = corpus.get_generator_func(corpus.TRAIN)
eval_data_generator_func = corpus.get_generator_func(corpus.EVAL)

corpus_meta_data = corpus.get_meta_info()

tags_data = generate_tagset(corpus_meta_data['tags'])

train_data = list(train_data_generator_func())
eval_data = list(eval_data_generator_func())

tag_lookuper = Lookuper({v: i for i, v in enumerate(tags_data)})
vocab_data_file = 'seq2annotation/data/unicode_char_list.txt'
vocabulary_lookuper = index_table_from_file(vocab_data_file)


def preprocss(data):
    raw_x = []
    raw_y = []

    for offset_data in data:
        tags = offset_to_biluo(offset_data)
        words = offset_data.text

        tag_ids = [tag_lookuper.lookup(i) for i in tags]
        word_ids = [vocabulary_lookuper.lookup(i) for i in words]

        raw_x.append(word_ids)
        raw_y.append(tag_ids)

    maxlen = max(len(s) for s in raw_x)

    x = tf.keras.preprocessing.sequence.pad_sequences(raw_x, maxlen,
                                                      padding='post')  # right padding

    # lef padded with -1. Indeed, any integer works as it will be masked
    # y_pos = pad_sequences(y_pos, maxlen, value=-1)
    # y_chunk = pad_sequences(y_chunk, maxlen, value=-1)
    y = tf.keras.preprocessing.sequence.pad_sequences(raw_y, maxlen, value=0,
                                                      padding='post')

    return x, y


transition_contrain = allowed_transitions(constraint_type.BIOUL, tag_lookuper.inverse_index_table)

train_x, train_y = preprocss(train_data)
test_x, test_y = preprocss(eval_data)

EPOCHS = 1
EMBED_DIM = 64
BiRNN_UNITS = 200

vacab_size = vocabulary_lookuper.size()
tag_size = tag_lookuper.size()

char_embed_layer = Embedding(vacab_size, EMBED_DIM, mask_zero=True)
char_bilstm_layer = Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True))(char_embed_layer)

dict_input_layer = Input(shape=)
dict_bilstm_layer = Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True))(dict_input_layer)

crf_layer = CRF(tag_size)

model.summary()

model.compile('adam', loss=crf_loss, metrics=[crf_accuracy])
model.fit(train_x, train_y, epochs=EPOCHS, validation_data=[test_x, test_y])

pred_y = model.predict(test_x)
test_y_pred = pred_y[test_x > 0]
test_y_true = test_y[test_x > 0]

print('\n---- Result of BiLSTM-CRF ----\n')
classification_report(test_y_true, test_y_pred, tags_data)
