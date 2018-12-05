import tensorflow as tf

from seq2annotation.algorithms.lookup_model import LookupModel


class BilstmCrfLookupModel(LookupModel):
    def bilstm_layer(self, embeddings, nwords):
        t = tf.transpose(embeddings, perm=[1, 0, 2])
        lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(self.params['lstm_size'])
        lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(self.params['lstm_size'])
        lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
        output_fw, _ = lstm_cell_fw(t, dtype=tf.float32,
                                    sequence_length=nwords)
        output_bw, _ = lstm_cell_bw(t, dtype=tf.float32,
                                    sequence_length=nwords)

        # lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.params['lstm_size'])
        # lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.params['lstm_size'])
        # outputs, output_fw, output_bw = tf.nn.bidirectional_dynamic_rnn(
        #     lstm_fw_cell,
        #     lstm_bw_cell, t,
        #     sequence_length=nwords,
        #     dtype=tf.float32
        # )

        output = tf.concat([output_fw, output_bw], axis=-1)
        # transpose it back
        output = tf.transpose(output, perm=[1, 0, 2])

        return output

    def call(self, embeddings, nwords):
        data = self.bilstm_layer(embeddings, nwords)
        return data
