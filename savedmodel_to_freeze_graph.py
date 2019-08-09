import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import graph_util

# bugfix; related to bug (about BlockLSTM): https://github.com/tensorflow/tensorflow/issues/23369
tf.contrib.rnn

export_dir = "/home/howl/workshop/seq2annotation_ner_on_ecarx/results/saved_model/BilstmCrfModel-64-0.001-None-15000/1564714194"

with tf.Session(graph=tf.Graph()) as sess:
    meta_graph_def = tf.saved_model.loader.load(sess, [tag_constants.SERVING], export_dir)

    output_tensor_name = meta_graph_def.signature_def['serving_default'].outputs['tags'].name

    output_node_name, _ = output_tensor_name.split(":")

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, [output_node_name, 'init_all_tables'])
    with tf.gfile.GFile('./model.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())
