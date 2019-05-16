import pkg_resources

import numpy as np

from seq2annotation.trainer.train_model_paddle import train_model
from seq2annotation import utils


class Model(object):
    def __init__(self, config):
        self.config = config

    def train_and_eval_then_save(self, train_inpf, eval_inpf, configure):
        evaluate_result, export_results, final_saved_model = train_model(
            train_inpf=train_inpf,
            eval_inpf=eval_inpf,
            config=configure
        )

        return evaluate_result, export_results, final_saved_model

    def get_default_config(self):
        data_dir = self.config.pop('data_dir', '.')
        result_dir = self.config.pop('result_dir', '.')

        params = {
            'dim': 300,
            'dropout': 0.5,
            'num_oov_buckets': 1,
            'epochs': 10,
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
            },

            'save_checkpoints_secs': 60,
            'learning_rate': 0.001,
            'max_steps': None,
            'max_steps_without_increase': 1000,
            'train_hook': {},
            'shuffle_pool_size': 30,
            'embedding_dim': 64,
        }

        vocab_data_file = pkg_resources.resource_filename(__name__,
                                                          './data/unicode_char_list.txt')
        params['vocab_data'] = np.loadtxt(vocab_data_file, dtype=np.unicode,
                                          encoding=None).tolist()

        # plus one for OOV
        params['embedding_vocabulary_size'] = len(params['vocab_data']) + 1

        return params
