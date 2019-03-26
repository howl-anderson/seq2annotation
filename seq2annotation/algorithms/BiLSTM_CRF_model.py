import tensorflow as tf
from seq2annotation.algorithms.model import Model


class BilstmCrfModel(Model):
    def bilstm_layer(self, embeddings, nwords):
        t = tf.transpose(embeddings, perm=[1, 0, 2])

        lstm_size = self.params['lstm_size']

        lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(lstm_size)
        lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(lstm_size)
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
        data = self.bilstm_layer(embeddings, nwords)
        return data
