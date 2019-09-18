import os
from collections import Counter

import numpy
import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, Bidirectional, LSTM
from tensorflow.python.keras.models import Sequential

from ioflow.configure import read_configure
from ioflow.corpus import get_corpus_processor
from seq2annotation.input import generate_tagset, Lookuper, \
    index_table_from_file
from tf_crf_layer.layer import CRF
from tf_crf_layer.loss import crf_loss, ConditionalRandomFieldLoss
from tf_crf_layer.metrics import crf_accuracy, SequenceCorrectness, SequenceSpanAccuracy, sequence_span_accuracy
from tokenizer_tools.tagset.converter.offset_to_biluo import offset_to_biluo

# tf.enable_eager_execution()


from seq2annotation import unrandom


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

vocab_data_file = os.path.join(os.path.dirname(__file__), '../data/unicode_char_list.txt')
vocabulary_lookuper = index_table_from_file(vocab_data_file)


def classification_report(y_true, y_pred, labels):
    """
    Similar to the one in sklearn.metrics,
    reports per classs recall, precision and F1 score
    """
    y_true = numpy.asarray(y_true).ravel()
    y_pred = numpy.asarray(y_pred).ravel()
    corrects = Counter(yt for yt, yp in zip(y_true, y_pred) if yt == yp)
    y_true_counts = Counter(y_true)
    y_pred_counts = Counter(y_pred)
    report = ((lab,  # label
               corrects[i] / max(1, y_true_counts[i]),  # recall
               corrects[i] / max(1, y_pred_counts[i]),  # precision
               y_true_counts[i]  # support
               ) for i, lab in enumerate(labels))
    report = [(l, r, p, 2 * r * p / max(1e-9, r + p), s) for l, r, p, s in report]

    print('{:<15}{:>10}{:>10}{:>10}{:>10}\n'.format('',
                                                    'recall',
                                                    'precision',
                                                    'f1-score',
                                                    'support'))
    formatter = '{:<15}{:>10.2f}{:>10.2f}{:>10.2f}{:>10d}'.format
    for r in report:
        print(formatter(*r))
    print('')
    report2 = list(zip(*[(r * s, p * s, f1 * s) for l, r, p, f1, s in report]))
    N = len(y_true)
    print(formatter('avg / total',
                    sum(report2[0]) / N,
                    sum(report2[1]) / N,
                    sum(report2[2]) / N, N) + '\n')


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

    print(">>> maxlen: {}".format(maxlen))

    x = tf.keras.preprocessing.sequence.pad_sequences(raw_x, maxlen,
                                                      padding='post')  # right padding

    # lef padded with -1. Indeed, any integer works as it will be masked
    # y_pos = pad_sequences(y_pos, maxlen, value=-1)
    # y_chunk = pad_sequences(y_chunk, maxlen, value=-1)
    y = tf.keras.preprocessing.sequence.pad_sequences(raw_y, maxlen, value=0,
                                                      padding='post')

    return x, y


train_x, train_y = preprocss(train_data)
test_x, test_y = preprocss(eval_data)

EPOCHS = config['epochs']
EMBED_DIM = config['embedding_dim']
BiRNN_UNITS = config['lstm_size']

vacab_size = vocabulary_lookuper.size()
tag_size = tag_lookuper.size()

model = Sequential()
model.add(Embedding(vacab_size, EMBED_DIM, mask_zero=True))
model.add(Bidirectional(LSTM(BiRNN_UNITS, return_sequences=True)))
model.add(CRF(tag_size, name='crf'))

# print model summary
model.summary()

callbacks_list = []

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=config['summary_log_dir'])
callbacks_list.append(tensorboard_callback)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(config['model_dir'], 'cp-{epoch:04d}.ckpt'),
    load_weights_on_restart=True,
    verbose=1
)
callbacks_list.append(checkpoint_callback)

metrics_list = []

metrics_list.append(crf_accuracy)
metrics_list.append(SequenceCorrectness())
metrics_list.append(sequence_span_accuracy)

loss_func = ConditionalRandomFieldLoss()
# loss_func = crf_loss

model.compile('adam', loss={'crf': loss_func}, metrics=metrics_list)
model.fit(
    train_x, train_y,
    epochs=EPOCHS,
    validation_data=[test_x, test_y],
    callbacks=callbacks_list
)

# Save the model
model.save(config['h5_model_file'])
tf.keras.experimental.export_saved_model(model, config['saved_model_dir'])
