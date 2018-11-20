from seq2annotation import train_model
from seq2annotation.algorithms.BiLSTM_CRF_model import BilstmCrfModel
from seq2annotation.algorithms.IDCNN_CRF_model import IdcnnCrfModel


# train_model(data_dir='./data', result_dir='./result', model_fn=IdcnnCrfModel.model_fn, **IdcnnCrfModel.default_params())
train_model(data_dir='./data', result_dir='./result', model_fn=BilstmCrfModel.model_fn, **BilstmCrfModel.default_params())