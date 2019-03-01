import os

import ray
import ray.tune as tune
from seq2annotation.trainer.train_model import train_model
from seq2annotation.algorithms.BiLSTM_CRF_model import BilstmCrfModel

ray.init()

cwd = os.getcwd()


def train_func(config, reporter):
    evaluate_result, export_results = train_model(
        data_dir=os.path.join(cwd, './data'), result_dir=os.path.join(cwd, './results'),
        train_spec={'max_steps': 150000},
        hook={
            'stop_if_no_increase': {
                'min_steps': 100,
                'run_every_secs': 60,
                'max_steps_without_increase': 10000
            }
        },
        model=BilstmCrfModel,
        optimizer_params={'learning_rate': config['momentum']},
        **BilstmCrfModel.default_params(),
    )

    reporter(mean_accuracy=evaluate_result['correct_rate'])


all_trials = tune.run_experiments({
    "my_experiment": {
        "run": train_func,
        "stop": {"mean_accuracy": 0.96},
        "config": {"momentum": tune.grid_search([0.05, 0.1, 0.2])}
    }
})

print(all_trials)
