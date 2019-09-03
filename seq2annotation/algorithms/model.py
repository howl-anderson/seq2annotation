from pathlib import Path

import tensorflow as tf
import numpy as np
from seq2annotation.metrics import precision, recall, f1
from tokenizer_tools.metrics import correct_rate


class Model(object):
    @classmethod
    def default_params(cls):
        return {}

    @classmethod
    def get_model_name(cls):
        return cls.__name__

    @classmethod
    def model_fn(cls, features, labels, mode, params):
        instance = cls(features, labels, mode, params)
        return instance()

    def __init__(self, features, labels, mode, params):
        self.features = features
        self.labels = labels
        self.mode = mode
        self.params = params

    def tpu_input_layer(self):
        word_ids = self.features["words"]

        nwords = tf.identity(self.features["words_len"], name="input_words_len")

        indices = self.params["_indices"]
        num_tags = self.params["_num_tags"]

        return indices, num_tags, word_ids, nwords

    def input_layer(self):
        # data = np.loadtxt(self.params['vocab'], dtype=np.unicode, encoding=None)
        data = self.params["vocab_data"]
        mapping_strings = tf.Variable(data)
        vocab_words = tf.contrib.lookup.index_table_from_tensor(
            mapping_strings, num_oov_buckets=1
        )

        # Word Embeddings
        words = tf.identity(self.features["words"], name="input_words")
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

        nwords = tf.identity(self.features["words_len"], name="input_words_len")

        # get tag info
        # with Path(self.params['tags']).open() as f:
        indices = [
            idx
            for idx, tag in enumerate(self.params["tags_data"])
            if tag.strip() != "O"
        ]
        num_tags = len(indices) + 1

        # # true tags to ids
        # if self.mode == tf.estimator.ModeKeys.PREDICT:
        #     true_tag_ids = 0
        # else:
        #     true_tag_ids = self.tag2id(self.labels)

        return indices, num_tags, word_ids, nwords

    def embedding_layer(self, word_ids):
        # load pre-trained data from file
        # glove = np.load(params['glove'])['embeddings']  # np.array

        # training the embedding during training
        glove = np.zeros(
            (self.params["embedding_vocabulary_size"], self.params["embedding_dim"]),
            dtype=np.float32,
        )

        # Add OOV word embedding
        embedding_array = np.vstack([glove, [[0.0] * self.params["embedding_dim"]]])

        embedding_variable = tf.Variable(
            embedding_array, dtype=tf.float32, trainable=True
        )
        embeddings = tf.nn.embedding_lookup(embedding_variable, word_ids)

        return embeddings

    def dropout_layer(self, data):
        training = self.mode == tf.estimator.ModeKeys.TRAIN
        output = tf.layers.dropout(data, rate=self.params["dropout"], training=training)

        return output

    def dense_layer(self, data, num_tags):
        logits = tf.layers.dense(data, num_tags)

        return logits

    def load_tag_data(self):
        # data = np.loadtxt(self.params['tags'], dtype=np.unicode, encoding=None)
        data = self.params["tags_data"]
        mapping_strings = tf.Variable(data)

        return mapping_strings

    def load_word_data(self):
        data = np.loadtxt(self.params["words"], dtype=np.unicode, encoding=None)
        mapping_strings = tf.Variable(data.reshape((-1,)))

        return mapping_strings

    def tag2id(self, labels, name=None):
        mapping_strings = self.load_tag_data()
        vocab_tags = tf.contrib.lookup.index_table_from_tensor(
            mapping_strings, name=name
        )

        tags = vocab_tags.lookup(labels)

        return tags

    def id2tag(self, pred_ids, name=None):
        mapping_strings = self.load_tag_data()
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_tensor(
            mapping_strings, name=name
        )

        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))

        return pred_strings

    def id2word(self, word_ids, name=None):
        mapping_strings = self.load_word_data()
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_tensor(
            mapping_strings, name=name
        )

        word_strings = reverse_vocab_tags.lookup(tf.to_int64(word_ids))

        return word_strings

    def loss_layer(self, preds, ground_true, nwords, crf_params):
        with tf.name_scope("CRF_log_likelihood"):
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                preds, ground_true, nwords, crf_params
            )

        loss = tf.reduce_mean(-log_likelihood)

        return loss

    def crf_decode_layer(self, logits, crf_params, nwords):
        with tf.name_scope("CRF_decode"):
            pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

        return pred_ids

    def compute_metrics(self, tags, pred_ids, num_tags, indices, nwords):
        weights = tf.sequence_mask(nwords)

        # metrics_correct_rate, golden, predict = correct_rate(tags, pred_ids)
        # metrics_correct_rate = correct_rate(tags, pred_ids, weights)

        metrics = {
            "acc": tf.metrics.accuracy(tags, pred_ids, weights),
            "precision": precision(tags, pred_ids, num_tags, indices, weights),
            "recall": recall(tags, pred_ids, num_tags, indices, weights),
            "f1": f1(tags, pred_ids, num_tags, indices, weights),
            "correct_rate": correct_rate(tags, pred_ids, weights),
            # 'golden': (golden, tf.zeros([], tf.int32)),
            # 'predict': (predict, tf.zeros([], tf.int32))
        }

        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        return metrics

    def call(self, embeddings, nwords):
        raise NotImplementedError

    def __call__(self):
        indices, num_tags, word_ids, nwords = self.input_layer()
        # indices, num_tags, word_ids, nwords = self.tpu_input_layer()

        embeddings = self.embedding_layer(word_ids)

        data = self.call(embeddings, nwords)

        data = self.dropout_layer(data)
        logits = self.dense_layer(data, num_tags)

        crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)

        pred_ids = self.crf_decode_layer(logits, crf_params, nwords)
        pred_strings = self.id2tag(pred_ids, name="predict")

        # word_strings = self.id2word(word_ids, name='word_strings')

        # print(word_strings)

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {"pred_ids": pred_ids, "tags": pred_strings}

            if self.params["use_tpu"]:
                return tf.contrib.tpu.TPUEstimatorSpec(
                    self.mode, predictions=predictions
                )
            else:
                return tf.estimator.EstimatorSpec(self.mode, predictions=predictions)
        else:
            # true_tag_ids = self.labels
            true_tag_ids = self.tag2id(self.labels, "labels")

            # print(pred_strings)
            # print(self.labels)

            loss = self.loss_layer(logits, true_tag_ids, nwords, crf_params)

            metrics = self.compute_metrics(
                true_tag_ids, pred_ids, num_tags, indices, nwords
            )

            if self.mode == tf.estimator.ModeKeys.EVAL:
                if self.params["use_tpu"]:

                    def my_metric_fn(true_tag_ids, pred_ids, num_tags, indices, nwords):
                        return self.compute_metrics(
                            true_tag_ids, pred_ids, num_tags, indices, nwords
                        )

                    return tf.contrib.tpu.TPUEstimatorSpec(
                        self.mode,
                        loss=loss,
                        eval_metrics=(
                            my_metric_fn,
                            [true_tag_ids, pred_ids, num_tags, indices, nwords],
                        ),
                    )
                else:
                    # set up metrics reporter
                    # from ioflow.hooks.metrics_reporter import metrics_report_hook
                    # hook_class = metrics_report_hook(self.params)
                    # hook_metrics = {"eval_loss": loss}
                    # hook_metrics.update({'_'.join(['eval', k]): v[0] for k, v in metrics.items()})
                    # hook = hook_class(hook_metrics)

                    # return tf.estimator.EstimatorSpec(
                    #     self.mode, loss=loss, eval_metric_ops=metrics, evaluation_hooks=[hook])

                    return tf.estimator.EstimatorSpec(
                        self.mode, loss=loss, eval_metric_ops=metrics
                    )

            elif self.mode == tf.estimator.ModeKeys.TRAIN:
                train_op = tf.train.AdamOptimizer(
                    **self.params.get("optimizer_params", {})
                ).minimize(loss, global_step=tf.train.get_or_create_global_step())
                if self.params["use_tpu"]:
                    train_op = tf.contrib.tpu.CrossShardOptimizer(train_op)

                if self.params["use_tpu"]:
                    return tf.contrib.tpu.TPUEstimatorSpec(
                        self.mode, loss=loss, train_op=train_op
                    )
                else:
                    # set up metrics reporter
                    # from ioflow.hooks.metrics_reporter import metrics_report_hook
                    # hook_class = metrics_report_hook(self.params)
                    # hook = hook_class({"train_loss": loss})

                    # return tf.estimator.EstimatorSpec(
                    #     self.mode, loss=loss, train_op=train_op, training_hooks=[hook])

                    return tf.estimator.EstimatorSpec(
                        self.mode, loss=loss, train_op=train_op
                    )
