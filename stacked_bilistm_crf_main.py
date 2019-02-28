from seq2annotation.trainer.train_model import train_model
from seq2annotation.algorithms.Stacked_BiLSTM_CRF_model import StackedBilstmCrfModel


# train_model(data_dir='./data', result_dir='./result', model_fn=IdcnnCrfModel.model_fn, **IdcnnCrfModel.default_params())
train_model(
    data_dir='./data', result_dir='./results',
    train_spec={'max_steps': 150000},
    hook={
        'stop_if_no_increase': {
            'min_steps': 100,
            'run_every_secs': 60,
            'max_steps_without_increase': 10000
        }
    },
    model=StackedBilstmCrfModel, **StackedBilstmCrfModel.default_params()
)
