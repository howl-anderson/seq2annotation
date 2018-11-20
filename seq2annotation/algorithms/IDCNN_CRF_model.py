import tensorflow as tf
from seq2annotation.algorithms.model import Model


class IdcnnCrfModel(Model):
    @classmethod
    def default_params(cls):
        default_params = {
            'filter_width': 2,
            'num_filter': 100,
            'repeat_times': 4,
            'layers': [
                {
                    'dilation': 1,
                },
                {
                    'dilation': 1,
                },
                {
                    'dilation': 2,
                }
            ]
        }

        return default_params

    def idcnn_layer(self, embeddings):
        with tf.variable_scope("idcnn"):
            model_inputs = tf.expand_dims(embeddings, 1)

            shape = [1, self.params['filter_width'], self.params['dim'], self.params['num_filter']]
            print(shape)
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=shape)

            """
            shape of input = [batch, in_height, in_width, in_channels]
            shape of filter = [filter_height, filter_width, in_channels, out_channels]
            """
            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer")
            finalOutFromLayers = []
            totalWidthForLastDim = 0
            for j in range(self.params['repeat_times']):
                for i in range(len(self.params['layers'])):
                    dilation = self.params['layers'][i]['dilation']
                    isLast = True if i == (
                                len(self.params['layers']) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=tf.AUTO_REUSE):
                        w = tf.get_variable(
                            "filterW",
                            shape=[1, self.params['filter_width'],
                                   self.params['num_filter'], self.params['num_filter']],
                            initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("filterB",
                                            shape=[self.params['num_filter']])
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.params['num_filter']
                        layerInput = conv
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            finalOut = tf.squeeze(finalOut, [1])

        return finalOut

    def call(self, embeddings, nwords):
        data = self.idcnn_layer(embeddings)
        return data
