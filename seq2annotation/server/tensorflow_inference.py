from typing import List

import keras

from tokenizer_tools.tagset.NER.BILUO import BILUOSequenceEncoderDecoder
from tokenizer_tools.tagset.offset.sequence import Sequence
from tensorflow.contrib import predictor

from tokenizer_tools.tagset.exceptions import TagSetDecodeError

decoder = BILUOSequenceEncoderDecoder()


class Inference(object):
    def __init__(self, model_path):
        # load model
        self.model_dir = model_path
        self.predict_fn = predictor.from_saved_model(model_path)

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

        # TODO: batch infer will cause padding, which will maybe cause decoder to offset bug.
        # TODO: feature translate should out of this main program for better compatible with keras and estimator model
        input_feature = {
            'words': [[i for i in text] for text in sentence],
            'words_len': [len(text) for text in raw_sequences],
        }

        # print(input_feature)

        predictions = self.predict_fn(input_feature)
        tags_list = predictions['tags']

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
