from typing import List

import keras
import tensorflow as tf

from tokenizer_tools.tagset.NER.BILUO import BILUOSequenceEncoderDecoder
from tokenizer_tools.tagset.offset.sequence import Sequence

from tokenizer_tools.tagset.exceptions import TagSetDecodeError

decoder = BILUOSequenceEncoderDecoder()

from tf_crf_layer.metrics.crf_accuracy import crf_accuracy
from tf_crf_layer.metrics.sequence_span_accuracy import sequence_span_accuracy


class Inference(object):
    def __init__(self, model_path):
        # load model
        self.model_dir = model_path

        # TODO: temp bugfix
        self.model = tf.keras.models.load_model(model_path, custom_objects={"crf_accuracy": crf_accuracy, "sequence_span_accuracy": sequence_span_accuracy})
        self.predict_fn = self.model.predict

    def infer(self, input_text: str):
        infer_result = self._infer(input_text)
        return infer_result[0]

    def batch_infer(self, input_text: List[str]):
        return self._infer(input_text)

    def _infer(self, input_text):
        if isinstance(input_text, str):
            input_list = [input_text]
        else:
            input_list = input_text

        raw_sequences = [[i for i in text] for text in input_list]

        sentence = keras.preprocessing.sequence.pad_sequences(
            raw_sequences, dtype='object',
            padding='post', truncating='post', value=['<pad>']
        ).tolist()

        tags_list = self.predict_fn(sentence)

        infer_result = []
        for raw_input_text, raw_text, normalized_text, tags in zip(input_list, raw_sequences, sentence, tags_list):
            # decode Unicode
            tags_seq = [i.decode() for i in tags]

            # print(tags_seq)

            # BILUO to offset
            failed = False
            try:
                seq = decoder.to_offset(tags_seq, raw_text)
            except TagSetDecodeError as e:
                print(e)

                # invalid tag sequence will raise exception
                # so return a empty result
                seq = Sequence(input_text)
                failed = True

            infer_result.append((raw_input_text, seq, tags_seq, failed))

        return infer_result


if __name__ == "__main__":
    pass
