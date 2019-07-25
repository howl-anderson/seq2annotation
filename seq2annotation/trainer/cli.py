import json

from ioflow.corpus import get_corpus_processor
from ioflow.eval_reporter import get_eval_reporter
from ioflow.task_status import get_task_status
from ioflow.model_saver import get_model_saver
from ioflow.configure import read_configure

from seq2annotation.input import build_input_func, generate_tagset
from seq2annotation.model import Model
from seq2annotation.health_check_transponder import run_health_check_transponder_in_background

# start health check
run_health_check_transponder_in_background()

raw_config = read_configure()
model = Model(raw_config)

config = model.get_default_config()
config.update(raw_config)

task_status = get_task_status(config)

task_status.send_status(task_status.START)

# read data according configure
# # report status: start to download corpus
task_status.send_status(task_status.START_DOWNLOAD_CORPUS)

corpus = get_corpus_processor(config)
corpus.prepare()
train_data_generator_func = corpus.get_generator_func(corpus.TRAIN)
eval_data_generator_func = corpus.get_generator_func(corpus.EVAL)

corpus_meta_data = corpus.get_meta_info()

# config['vocab_data'] = corpus_meta_data['vocab']
# vocab_data_file = pkg_resources.resource_filename(__name__, '../data/unicode_char_list.txt')
# config['vocab_data'] = np.loadtxt(vocab_data_file, dtype=np.unicode, encoding=None)

config['tags_data'] = generate_tagset(corpus_meta_data['tags'])

# build model according configure


# # report status: start to process corpus
task_status.send_status(task_status.START_PROCESS_CORPUS)

# train and evaluate model
train_input_func = build_input_func(train_data_generator_func, config)
eval_input_func = build_input_func(eval_data_generator_func, config) if eval_data_generator_func else None

# ***** test ******
# train_iterator = train_input_func()
import tensorflow as tf
# import sys

# for i, data in enumerate(train_data_generator_func()):
#     print(i, data)
#
#
# with tf.Session() as sess:
#     sess.run(tf.tables_initializer())
#
#     counter = 0
#     while True:
#         try:
#             value = sess.run(train_iterator)
#             counter += 1
#             print(value)
#             break
#         except tf.errors.OutOfRangeError:
#             break
#
# print(counter)
# #
# sys.exit(0)
# ***** /test ******

# # report status: start to train
task_status.send_status(task_status.START_TRAIN)
task_status.send_progress(0)
task_status.send_progress(50)

evaluate_result, export_results, final_saved_model = model.train_and_eval_then_save(
    train_input_func,
    eval_input_func,
    config
)

task_status.send_progress(100)

# # report status: start to test
task_status.send_status(task_status.START_TEST)

from seq2annotation.server.tensorflow_inference import Inference
from seq2annotation.server.http import sequence_to_response

eval_reporter = get_eval_reporter(config)

# result_list = []
inference = Inference(final_saved_model)
for item in eval_data_generator_func():
    text, result, _, _ = inference.infer(''.join(item.text))
    eval_reporter.record_x_and_y(item, sequence_to_response(text, result))
    # result_list.append(result)

eval_reporter.submit()

# if evaluate_result:
#     performance_metrics = get_performance_reporter(config)
#     performance_metrics.log_performance('test_loss', evaluate_result['loss'])
#     performance_metrics.log_performance('test_acc', evaluate_result['acc'])

# # report status: start to upload model
task_status.send_status(task_status.START_UPLOAD_MODEL)

model_saver = get_model_saver(config)
model_saver.save_model(final_saved_model)

task_status.send_status(task_status.DONE)
