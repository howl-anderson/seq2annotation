from seq2annotation.trainer.train_model import train_model
from seq2annotation.algorithms.BiLSTM_CRF_model import BilstmCrfModel
from seq2annotation.algorithms.IDCNN_CRF_model import IdcnnCrfModel


# train_model(data_dir='./data', result_dir='./result', model_fn=IdcnnCrfModel.model_fn, **IdcnnCrfModel.default_params())
result = train_model(
    data_dir='./data', result_dir='./results',
    train_spec={'max_steps': None},
    hook={
        'stop_if_no_increase': {
            'min_steps': 100,
            'run_every_secs': 60,
            'max_steps_without_increase': 10000
        }
    },
    use_gpu=True,
    tpu_config={
        'tpu_name': 'u1mail2me',
    },
    model=BilstmCrfModel, **BilstmCrfModel.default_params()
)

print(result)
