import copy
import functools
import json
from pathlib import Path

import tensorflow as tf

# from seq2annotation.data_input.simple import input_fn as simple_input_fn
from seq2annotation.data_input.with_lookup import input_fn as simple_input_fn
from seq2annotation.data_input.simple import generator_fn as simple_generator_fn
from seq2annotation.algorithms.BiLSTM_CRF_model import BilstmCrfModel

tf.logging.set_verbosity(tf.logging.INFO)


def train_model(**kwargs):
    data_dir = kwargs.pop('data_dir', '.')
    result_dir = kwargs.pop('result_dir', '.')
    input_fn = kwargs.pop('input_fn', simple_input_fn)
    generator_fn = kwargs.pop('generator_fn', simple_generator_fn)
    model_fn = kwargs.pop('model_fn', BilstmCrfModel.model_fn)

    params = {
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1,
        'epochs': None,
        'batch_size': 20,
        'buffer': 15000,
        'lstm_size': 100,
        'words': str(Path(data_dir, './unicode_char_list.txt')),
        'lookup': str(Path(data_dir, './lookup.txt')),
        'chars': str(Path(data_dir, 'vocab.chars.txt')),
        'tags': str(Path(data_dir, './tags.txt')),
        'glove': str(Path(data_dir, './glove.npz')),

        'model_dir': str(Path(result_dir, 'model_dir')),
        'params_log_file': str(Path(result_dir, 'params.json')),

        'train': str(Path(data_dir, '{}.txt'.format('train'))),
        'test': str(Path(data_dir, '{}.txt'.format('test'))),

        'preds': {
            'train': str(Path(result_dir, '{}.txt'.format('preds_train'))),
            'test': str(Path(result_dir, '{}.txt'.format('preds_test'))),
        },

        'saved_model_dir': str(Path(result_dir, 'saved_model')),

        'hook': {
            'stop_if_no_increase': {
                'min_steps': 100,
                'run_every_secs': 60,
                'max_steps_without_increase': 20
            }
        },

        'train_spec': {
            'max_steps': 5000
        },
        'eval_spec': {
            'throttle_secs': 120
        },

        'estimator': {
            'save_checkpoints_secs': 120
        },


        'embedding': {
            'vocabulary_size': 128003
        }
    }

    # update from kwargs
    params.update(kwargs)

    with open(params['params_log_file'], 'w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    def fwords(name):
        return params[name]

    def preds_file(name):
        return params['preds'][name]

    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, fwords('train'),
                                   params, shuffle_and_repeat=True)
    eval_inpf = functools.partial(input_fn, fwords('test'))

    estimator_params = copy.deepcopy(params)
    # estimator_params.update({
    #     'words_feature_columns': words_feature_columns,
    #     'words_len_feature_columns': words_len_feature_columns
    # })

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=params['estimator']['save_checkpoints_secs'])

    estimator = tf.estimator.Estimator(model_fn, params['model_dir'], cfg, estimator_params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)

    hook_params = params['hook']['stop_if_no_increase']
    hook = tf.contrib.estimator.stop_if_no_increase_hook(
        estimator, 'f1',
        max_steps_without_increase=hook_params['max_steps_without_increase'],
        min_steps=hook_params['min_steps'],
        run_every_secs=hook_params['run_every_secs']
    )

    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook], max_steps=params['train_spec']['max_steps'])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=params['eval_spec']['throttle_secs'])
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    # estimator.train(input_fn=train_inpf, hooks=[hook])

    # Write predictions to file
    def write_predictions(name):
        output_file = preds_file(name)
        with open(output_file, 'wt') as f:
            test_inpf = functools.partial(input_fn, fwords(name))
            golds_gen = generator_fn(fwords(name))
            preds_gen = estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), tags) = golds
                preds_tags = [i.decode() for i in preds['tags']]
                for word, tag, tag_pred in zip(words, tags, preds_tags):
                    # f.write(b' '.join([word, tag, tag_pred]) + b'\n')
                    f.write(' '.join([word, tag, tag_pred]) + '\n')
                # f.write(b'\n')
                f.write('\n')

    for name in ['train', 'test']:
        write_predictions(name)

    # export saved_model
    feature_spec = {
        'words': tf.placeholder(tf.string, [None, None]),
        'words_len': tf.placeholder(tf.int32, [None]),
        'lookup': tf.placeholder(tf.float32, [None, None, 6])
    }
    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
    estimator.export_saved_model(
        params['saved_model_dir'],
        serving_input_receiver_fn,
    )
