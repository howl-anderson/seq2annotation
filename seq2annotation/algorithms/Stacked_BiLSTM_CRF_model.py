import tensorflow as tf
from seq2annotation.algorithms.model import Model


class StackedBilstmCrfModel(Model):
    @classmethod
    def default_params(cls):
        default_params = {
            'stacked_layers': 2
        }

        return default_params

    def bilstm_layer(self, embeddings, nwords):
        t = tf.transpose(embeddings, perm=[1, 0, 2])
        lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(self.params['lstm_size'])
        lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(self.params['lstm_size'])
        lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
        output_fw, _ = lstm_cell_fw(t, dtype=tf.float32,
                                    sequence_length=nwords)
        output_bw, _ = lstm_cell_bw(t, dtype=tf.float32,
                                    sequence_length=nwords)
        output = tf.concat([output_fw, output_bw], axis=-1)
        # transpose it back
        output = tf.transpose(output, perm=[1, 0, 2])

        return output

    def call(self, embeddings, nwords):
        inner_layer_data = self.bilstm_layer(embeddings, nwords)
        for i in range(1, self.params['stacked_layers']):
            inner_layer_data = self.bilstm_layer(inner_layer_data, nwords)

        return inner_layer_data
