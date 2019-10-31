import keras
from tensorflow.contrib import predictor

from seq2annotation.server.inference.inference_base import InferenceBase
from tokenizer_tools.tagset.NER.BILUO import BILUOSequenceEncoderDecoder

decoder = BILUOSequenceEncoderDecoder()


class TensorFlowInference(InferenceBase):
    def __init__(self, *args, **kwargs):
        super(TensorFlowInference, self).__init__(*args, **kwargs)

        self.raw_sequence = None

    def instance_predict_fn(self):
        return predictor.from_saved_model(self.model_path)

    def preprocess(self, input_):
        self.raw_sequence = input_

        sentence = keras.preprocessing.sequence.pad_sequences(
            input_, dtype="object", padding="post", truncating="post", value=["<pad>"]
        ).tolist()

        return sentence

    def encode_input_feature(self, msg_list):
        input_feature = {
            "words": [[i for i in text] for text in msg_list],
            "words_len": [len(text) for text in self.raw_sequence],
        }

        return input_feature

    def decode_output_feature(self, response):
        tags_list = response["tags"].tolist()

        return [[i.decode() for i in j] for j in tags_list]

    def postprocess(self, input_):
        return input_
