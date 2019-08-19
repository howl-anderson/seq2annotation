import tensorflow as tf
from tensorflow.python.platform import gfile

# only for bugfix
tf.contrib.rnn

output_graph_path = './model.pb'

graph = tf.Graph()

with gfile.FastGFile(output_graph_path, 'rb') as f:
    output_graph_def = tf.GraphDef()
    output_graph_def.ParseFromString(f.read())

with graph.as_default():
    tf.import_graph_def(output_graph_def, name="")

    with tf.Session(graph=graph) as sess:
        init_all_tables = graph.get_operation_by_name('init_all_tables')
        sess.run(init_all_tables)
        # sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())
        # 得到当前图有几个操作节点
        print("%d ops in the final graph." % len(output_graph_def.node))

        tensor_name = [tensor.name for tensor in output_graph_def.node]
        print(tensor_name)
        print('---------------------------')

        Placeholder = sess.graph.get_tensor_by_name('Placeholder:0')
        Placeholder_1 = sess.graph.get_tensor_by_name('Placeholder_1:0')
        # embedding层的输出
        embedding_out = sess.graph.get_tensor_by_name('embedding_lookup:0')
        enbedding_transpose = sess.graph.get_tensor_by_name('transpose:0')
        # BiLSTM层的输出
        BiLSTM_out = sess.graph.get_tensor_by_name('concat:0')
        BiLSTM_transpose_1 = sess.graph.get_tensor_by_name('transpose_1:0')

        a = sess.graph.get_tensor_by_name('Variable_1:0')
        a_array = a.eval(session=sess)
        for i in a_array[:1]:
            print(i)
        print('#####################')

        input_words = [['唱', '一', '首', '不', '消', '失', '的', '回', '忆']]
        input_words_len = [9]

        b = sess.graph.get_tensor_by_name('hash_table_Lookup/hash_table_Lookup/LookupTableFindV2:0')
        b = sess.run(b, {Placeholder: input_words, Placeholder_1: input_words_len})

        for i in b:
            print(i)