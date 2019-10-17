from tensorflow.python import keras

from seq2annotation.input import Lookuper
from seq2annotation.server.inference.inference_base import InferenceBase


class KerasInferenceBase(InferenceBase):
    def __init__(self, model_path, tag_lookup_file=None, vocabulary_lookup_file=None):
        self.tag_lookup_table = Lookuper.load_from_file(tag_lookup_file)
        self.vocabulary_lookup_table = Lookuper.load_from_file(vocabulary_lookup_file)

        super(KerasInferenceBase, self).__init__(model_path)

    def preprocess(self, input_text: str):
        id_sequences = self.vocabulary_lookup_table.lookup_list_of_str_list(input_text)

        sentence = keras.preprocessing.sequence.pad_sequences(
            id_sequences, dtype="object", padding="post", truncating="post", value=0
        )

        return sentence

    def encode_input_feature(self, input_):
        return input_

    def decode_output_feature(self, input_):
        return input_

    def postprocess(self, tags_id_list):
        tags_list = self.tag_lookup_table.inverse_lookup_list_of_id_list(tags_id_list)

        return tags_list
