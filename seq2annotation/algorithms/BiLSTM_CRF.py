from pathlib import Path

import numpy as np
import tensorflow as tf
from tf_metrics import precision, recall, f1


def model_fn(features, labels, mode, params):
    # Read vocabs and inputs
    dropout = params['dropout']

    # feature_column based approach
    # orig_words_feature = tf.identity(features['words'], name='input_words')
    # orig_words_feature_shape = tf.shape(orig_words_feature)
    # words_feature = tf.reshape(orig_words_feature, [-1])
    # raw_embeddings = tf.feature_column.input_layer({'words': words_feature}, params['words_feature_columns'])
    # embeddings_shape = [orig_words_feature_shape[0], orig_words_feature_shape[1], 300]
    # embeddings = tf.reshape(raw_embeddings, embeddings_shape)

    # vocab_words = tf.contrib.lookup.index_table_from_file(
    #     params['words'], num_oov_buckets=params['num_oov_buckets'])

    data = np.loadtxt(params['words'], dtype=np.unicode, encoding=None)
    mapping_strings = tf.Variable(data.reshape((-1,)))
    vocab_words = tf.contrib.lookup.index_table_from_tensor(
        mapping_strings, num_oov_buckets=params['num_oov_buckets'])

    # Word Embeddings
    words = tf.identity(features['words'], name='input_words')
    word_ids = vocab_words.lookup(words)
    # glove = np.load(params['glove'])['embeddings']  # np.array
    glove = np.zeros((128003, params['dim']), dtype=np.float32)

    # Add OOV word embedding
    variable = np.vstack([glove, [[0.]*params['dim']]])

    variable = tf.Variable(variable, dtype=tf.float32, trainable=True)
    embeddings = tf.nn.embedding_lookup(variable, word_ids)

    #
    # raw_nwords = tf.identity(features['words_len'], name='input_words_len')
    # nwords = tf.feature_column.input_layer({'words_len': raw_nwords}, params['words_len_feature_columns'])
    # nwords = tf.reshape(nwords, [-1])
    # nwords = tf.to_int32(nwords)

    # words = features['words']
    # words = tf.convert_to_tensor(words)
    #
    # nwords = features['words_len']
    # nwords = tf.convert_to_tensor(nwords)

    nwords = tf.identity(features['words_len'], name='input_words_len')

    training = (mode == tf.estimator.ModeKeys.TRAIN)
    # vocab_words = tf.contrib.lookup.index_table_from_file(
    #     params['words'], num_oov_buckets=params['num_oov_buckets'])
    with Path(params['tags']).open() as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1

    # # Word Embeddings
    # word_ids = vocab_words.lookup(words)
    # glove = np.load(params['glove'])['embeddings']  # np.array
    #
    # # Add OOV word embedding
    # variable = np.vstack([glove, [[0.]*params['dim']]])
    #
    # variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    # embeddings = tf.nn.embedding_lookup(variable, word_ids)
    # embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

    # LSTM

    # because LSTMBlockFusedCell requires shape [time_len, batch_size, input_size]
    t = tf.transpose(embeddings, perm=[1, 0, 2])
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    output = tf.concat([output_fw, output_bw], axis=-1)
    # transpose it back
    output = tf.transpose(output, perm=[1, 0, 2])
    output = tf.layers.dropout(output, rate=dropout, training=training)

    # CRF
    logits = tf.layers.dense(output, num_tags)
    crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)

    with tf.name_scope("CRF_decode"):
        raw_pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

    pred_ids = tf.identity(raw_pred_ids, name="output_pred_ids")

    # Predictions
    # reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
    #     params['tags'])

    data = np.loadtxt(params['tags'], dtype=np.unicode, encoding=None)
    mapping_strings = tf.Variable(data.reshape((-1,)))
    reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_tensor(mapping_strings)

    pred_strings = tf.identity(
        reverse_vocab_tags.lookup(tf.to_int64(pred_ids)),
        name="output_pred_strings"
    )
    predictions = {
        'pred_ids': pred_ids,
        'tags': pred_strings
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        # vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
        data = np.loadtxt(params['tags'], dtype=np.unicode, encoding=None)
        mapping_strings = tf.Variable(data.reshape((-1,)))
        vocab_tags = tf.contrib.lookup.index_table_from_tensor(mapping_strings)

        tags = vocab_tags.lookup(labels)

        with tf.name_scope("CRF_log_likelihood"):
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                logits, tags, nwords, crf_params)

        loss = tf.reduce_mean(-log_likelihood)

        # Metrics
        weights = tf.sequence_mask(nwords)
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred_ids, weights),
            'precision': precision(tags, pred_ids, num_tags, indices, weights),
            'recall': recall(tags, pred_ids, num_tags, indices, weights),
            'f1': f1(tags, pred_ids, num_tags, indices, weights),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer().minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)