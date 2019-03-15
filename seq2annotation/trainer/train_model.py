import copy
import functools
import json
import os

import tensorflow as tf

from seq2annotation import utils
from seq2annotation.data_input.simple import input_fn as simple_input_fn
# from seq2annotation.data_input.with_lookup import input_fn as simple_input_fn
from seq2annotation.data_input.simple import generator_fn as simple_generator_fn
from seq2annotation.algorithms.BiLSTM_CRF_model import BilstmCrfModel
from tokenizer_tools.hooks import TensorObserveHook


tf.logging.set_verbosity(tf.logging.INFO)

observer_hook = TensorObserveHook(
    {
        'fake_golden': 'fake_golden:0',
        'fake_prediction': 'fake_prediction:0'
    },
    {
        "word_str": "word_strings_Lookup:0",
        "predictions_id": "predictions:0",
        "predict_str": "predict_Lookup:0",
        "labels_id": "labels:0",
        "labels_str": "IteratorGetNext:2",
    },
    {
        "word_str": lambda x: x.decode(),
        'predict_str': lambda x: x.decode(),
        'labels_str': lambda x: x.decode()
    }
)


def train_model(**kwargs):
    data_dir = kwargs.pop('data_dir', '.')
    result_dir = kwargs.pop('result_dir', '.')
    input_fn = kwargs.pop('input_fn', simple_input_fn)
    generator_fn = kwargs.pop('generator_fn', simple_generator_fn)
    model = kwargs.pop('model', None)
    model_name = kwargs.pop('model_name', None)
    model_fn = kwargs.pop('model_fn') if kwargs.get('model_fn') else getattr(model, 'model_fn')

    params = {
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1,
        'epochs': None,
        'batch_size': 20,
        'buffer': 15000,
        'lstm_size': 100,
        'words': utils.join_path(data_dir, './unicode_char_list.txt'),
        'lookup': utils.join_path(data_dir, './lookup.txt'),
        'chars': utils.join_path(data_dir, 'vocab.chars.txt'),
        'tags': utils.join_path(data_dir, './tags.txt'),
        'glove': utils.join_path(data_dir, './glove.npz'),

        'model_dir': utils.join_path(result_dir, 'model_dir'),
        'params_log_file': utils.join_path(result_dir, 'params.json'),

        'train': utils.join_path(data_dir, '{}.conllz'.format('train')),
        'test': utils.join_path(data_dir, '{}.conllz'.format('test')),

        'preds': {
            'train': utils.join_path(result_dir, '{}.txt'.format('preds_train')),
            'test': utils.join_path(result_dir, '{}.txt'.format('preds_test')),
        },

        'optimizer_params': {},

        'saved_model_dir': utils.join_path(result_dir, 'saved_model'),

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
            'throttle_secs': 60
        },

        'estimator': {
            'save_checkpoints_secs': 120
        },


        'embedding': {
            'vocabulary_size': 128003
        },

        'use_tpu': False,
        'tpu_config': {
            'tpu_name': None,
            'zone': None,
            'gcp_project': None
        }
    }

    # update from kwargs
    params.update(kwargs)

    with tf.io.gfile.GFile(params['params_log_file'], 'w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    def fwords(name):
        return params[name]

    def preds_file(name):
        return params['preds'][name]

    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, input_file=fwords('train'),
                                   config=params, shuffle_and_repeat=True)
    eval_inpf = functools.partial(input_fn, input_file=fwords('test'))

    estimator_params = copy.deepcopy(params)
    # estimator_params.update({
    #     'words_feature_columns': words_feature_columns,
    #     'words_len_feature_columns': words_len_feature_columns
    # })

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=params['estimator']['save_checkpoints_secs'])

    model_specific_name = '{model_name}-{batch_size}-{learning_rate}-{max_steps}-{max_steps_without_increase}'.format(
        model_name=model_name if model_name else model.get_model_name(),
        batch_size=params['batch_size'],
        learning_rate=params['optimizer_params'].get('learning_rate'),
        max_steps=params['train_spec'].get('max_steps'),
        max_steps_without_increase=params['hook'].get('max_steps_without_increase')
    )

    instance_model_dir = os.path.join(params['model_dir'], model_specific_name)

    if params['use_tpu']:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu=params['tpu_config']['tpu_name'],
            zone=params['tpu_config']['zone'],
            project=params['tpu_config']['gcp_project']
        )

        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=instance_model_dir,
            session_config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True),
            tpu_config=tf.contrib.tpu.TPUConfig(),
        )

        tpu_estimator_params = copy.deepcopy(estimator_params)
        # remove reserved keys
        # tpu_estimator_params['train_batch_size'] = tpu_estimator_params['batch_size']
        del tpu_estimator_params['batch_size']
        # del tpu_estimator_params['context']

        estimator = tf.contrib.tpu.TPUEstimator(model_fn=model_fn, params=tpu_estimator_params, config=run_config, use_tpu=True,
                                                train_batch_size=estimator_params['batch_size'],
                                                eval_batch_size=estimator_params['batch_size'],
                                                predict_batch_size=estimator_params['batch_size']
                                                )
    else:
        estimator = tf.estimator.Estimator(model_fn, instance_model_dir, cfg, estimator_params)


    # Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    utils.create_dir_if_needed(estimator.eval_dir())


    hook_params = params['hook']['stop_if_no_increase']
    hook = tf.contrib.estimator.stop_if_no_increase_hook(
        estimator, 'f1',
        max_steps_without_increase=hook_params['max_steps_without_increase'],
        min_steps=hook_params['min_steps'],
        run_every_secs=hook_params['run_every_secs']
    )

    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook], max_steps=params['train_spec']['max_steps'])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=params['eval_spec']['throttle_secs'], hooks=[observer_hook])
    evaluate_result, export_results = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    # estimator.train(input_fn=train_inpf, hooks=[hook])

    # Write predictions to file
    def write_predictions(name):
        output_file = preds_file(name)
        with tf.io.gfile.GFile(output_file, 'wt') as f:
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
    }

    instance_saved_dir = os.path.join(params['saved_model_dir'], model_specific_name)

    utils.create_dir_if_needed(instance_saved_dir)

    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
    estimator.export_saved_model(
        instance_saved_dir,
        serving_input_receiver_fn,
    )

    return evaluate_result, export_results
