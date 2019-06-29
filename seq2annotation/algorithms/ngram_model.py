from pathlib import Path

import tensorflow as tf
import numpy as np
from seq2annotation.metrics import precision, recall, f1
from tokenizer_tools.tagset.NER.BILUO import BILUOEncoderDecoder


class NgramModel(object):
    @classmethod
    def default_params(cls):
        return {}

    @classmethod
    def model_fn(cls, features, labels, mode, params):
        instance = cls(features, labels, mode, params)
        return instance()

    def __init__(self, features, labels, mode, params):
        self.features = features
        self.labels = labels
        self.mode = mode
        self.params = params

    def input_layer(self):
        data = np.loadtxt(self.params['words'], dtype=np.unicode, encoding=None)
        mapping_strings = tf.Variable(data.reshape((-1,)))
        vocab_words = tf.contrib.lookup.index_table_from_tensor(
            mapping_strings, num_oov_buckets=self.params['num_oov_buckets'])

        # Word Embeddings
        words = tf.identity(self.features['words'], name='input_words')
        word_ids = vocab_words.lookup(words)


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

        nwords = tf.identity(self.features['words_len'], name='input_words_len')

        # get tag info
        with Path(self.params['tags']).open() as f:
            indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
            num_tags = len(indices) + 1

        # # true tags to ids
        # if self.mode == tf.estimator.ModeKeys.PREDICT:
        #     true_tag_ids = 0
        # else:
        #     true_tag_ids = self.tag2id(self.labels)

        return indices, num_tags, word_ids, nwords

    def input_lookup_layer(self):
        # data = np.loadtxt(self.params['lookup'], dtype=np.unicode, encoding=None)
        # mapping_strings = tf.Variable(data.reshape((-1,)))
        # vocab_words = tf.contrib.lookup.index_table_from_tensor(
        #     mapping_strings, num_oov_buckets=self.params['num_oov_buckets'])
        #
        # # Word Embeddings
        # words = tf.identity(self.features['lookup'], name='lookup')
        # word_ids = vocab_words.lookup(words)
        #
        # return word_ids

        raw_feature = self.features['lookup']

        raw_feature_shape = tf.shape(raw_feature)
        flat_feature = tf.reshape(raw_feature, [-1])

        feature_column = tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                key='lookup',
                vocabulary_list=[0, 1])
        )

        encoded_feature = tf.feature_column.input_layer({'lookup': flat_feature}, [feature_column])

        raw_encoded_feature = tf.shape(encoded_feature)

        feature_shape = [raw_feature_shape[0], raw_feature_shape[1], 5]

        feature = tf.reshape(encoded_feature, feature_shape)

        return feature

    def embedding_layer(self, word_ids):
        # load pre-trained data from file
        # glove = np.load(params['glove'])['embeddings']  # np.array

        # training the embedding during training
        glove = np.zeros((self.params['embedding']['vocabulary_size'], self.params['dim']), dtype=np.float32)

        # Add OOV word embedding
        embedding_array = np.vstack([glove, [[0.] * self.params['dim']]])

        embedding_variable = tf.Variable(embedding_array, dtype=tf.float32, trainable=True)
        embeddings = tf.nn.embedding_lookup(embedding_variable, word_ids)

        return embeddings

    def dropout_layer(self, data):
        training = (self.mode == tf.estimator.ModeKeys.TRAIN)
        output = tf.layers.dropout(data, rate=self.params['dropout'], training=training)

        return output

    def dense_layer(self, data, num_tags):
        logits = tf.layers.dense(data, num_tags)

        return logits

    def load_tag_data(self):
        data = np.loadtxt(self.params['tags'], dtype=np.unicode, encoding=None)
        mapping_strings = tf.Variable(data.reshape((-1,)))

        return mapping_strings

    def tag2id(self, labels):
        mapping_strings = self.load_tag_data()
        vocab_tags = tf.contrib.lookup.index_table_from_tensor(mapping_strings)

        tags = vocab_tags.lookup(labels)

        return tags

    def id2tag(self, pred_ids):
        mapping_strings = self.load_tag_data()
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_tensor(
            mapping_strings)

        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))

        return pred_strings

    def loss_layer(self, preds, ground_true, nwords, crf_params):
        with tf.name_scope("CRF_log_likelihood"):
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                preds, ground_true, nwords, crf_params)

        loss = tf.reduce_mean(-log_likelihood)

        return loss

    def crf_decode_layer(self, logits, crf_params, nwords):
        with tf.name_scope("CRF_decode"):
            pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params,
                                                        nwords)

        return pred_ids

    def compute_metrics(self, tags, pred_ids, num_tags, indices, nwords):
        weights = tf.sequence_mask(nwords)
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred_ids, weights),
            'precision': precision(tags, pred_ids, num_tags, indices,
                                   weights),
            'recall': recall(tags, pred_ids, num_tags, indices, weights),
            'f1': f1(tags, pred_ids, num_tags, indices, weights),
        }

        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        return metrics

    def call(self, embeddings, nwords):
        raise NotImplementedError

    def __call__(self):
        indices, num_tags, word_ids, nwords = self.input_layer()
        char_embeddings = self.embedding_layer(word_ids)

        ngram_feature = self.features['lookup']

        char_feature_data = self.call(char_embeddings, nwords)
        ngram_feature_data = self.call(ngram_feature, nwords)

        feature_data = tf.concat((char_feature_data, ngram_feature_data), axis=2)

        feature_data_droped = self.dropout_layer(feature_data)
        logits = self.dense_layer(feature_data_droped, num_tags)

        crf_params = tf.get_variable("crf", [num_tags, num_tags],
                                     dtype=tf.float32)

        pred_ids = self.crf_decode_layer(logits, crf_params, nwords)

        pred_strings = self.id2tag(pred_ids)
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(self.mode,
                                              predictions=predictions)
        else:
            true_tag_ids = self.tag2id(self.labels)

            loss = self.loss_layer(logits, true_tag_ids, nwords, crf_params)

            metrics = self.compute_metrics(true_tag_ids, pred_ids,
                                           num_tags, indices, nwords)

            if self.mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    self.mode, loss=loss, eval_metric_ops=metrics)

            elif self.mode == tf.estimator.ModeKeys.TRAIN:
                train_op = tf.train.AdamOptimizer().minimize(
                    loss, global_step=tf.train.get_or_create_global_step())
                return tf.estimator.EstimatorSpec(
                    self.mode, loss=loss, train_op=train_op)
