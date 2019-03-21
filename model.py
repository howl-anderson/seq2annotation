from seq2annotation.trainer.train_model import train_model
from seq2annotation.algorithms.BiLSTM_CRF_model import BilstmCrfModel


class Model(object):
    def __init__(self, arch_setting):
        pass

    def train_and_eval_then_save(self, train_inpf, eval_inpf, saved_model_dir):
        evaluate_result, export_results, final_saved_model = train_model(
            train_inpf=train_inpf,
            eval_inpf=eval_inpf,
            forced_saved_model_dir=saved_model_dir,
            data_dir='./data', result_dir='./results',
            train_spec={'max_steps': 1500},
            hook={
                'stop_if_no_increase': {
                    'min_steps': 100,
                    'run_every_secs': 60,
                    'max_steps_without_increase': 10000
                }
            },
            model=BilstmCrfModel, **BilstmCrfModel.default_params()
        )

        return evaluate_result, export_results, final_saved_model
