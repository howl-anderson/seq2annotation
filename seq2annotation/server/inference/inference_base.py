from typing import List, Union, Any, Callable

import numpy as np

from tokenizer_tools.tagset.NER.BILUO import BILUOSequenceEncoderDecoder
from tokenizer_tools.tagset.offset.sequence import Sequence

from tokenizer_tools.tagset.exceptions import TagSetDecodeError

decoder = BILUOSequenceEncoderDecoder()


class InferenceBase(object):
    def __init__(self, model_path, *args, **kwargs):
        self.model_path = model_path

        self.predict_fn = None
        self.load_prediction_fn()

    def load_prediction_fn(self):
        self.predict_fn = self.instance_predict_fn()

    def instance_predict_fn(self) -> Callable:
        raise NotImplementedError

    def infer(self, input_text: str):
        input_text_list = [input_text]
        infer_result = self.do_infer(input_text_list)
        return infer_result[0]

    def batch_infer(self, input_text: List[str]):
        return self.do_infer(input_text)

    def preprocess(
        self, msg_list: List[List[str]]
    ) -> Union[np.ndarray, List[List[str]]]:
        raise NotImplementedError

    def encode_input_feature(self, msg_list: Union[np.ndarray, List[List[str]]]) -> Any:
        raise NotImplementedError

    def decode_output_feature(self, response: Any) -> Any:
        raise NotImplementedError

    def postprocess(self, input_: Any) -> List[List[str]]:
        raise NotImplementedError

    def decode_ner_tag_sequence(
        self,
        ner_tag_sequence: List[List[str]],
        std_msg_list: List[List[str]],
        raw_msg_list: List[str],
    ):
        infer_result = []
        for raw_msg, std_msg, ner_tags in zip(
            raw_msg_list, std_msg_list, ner_tag_sequence
        ):
            failed = False
            try:
                # BILUO to offset
                seq = decoder.to_offset(ner_tags, std_msg)
            except TagSetDecodeError as e:
                print(e)

                # invalid tag sequence will raise exception
                # so return a empty result
                seq = Sequence(std_msg)
                failed = True

            infer_result.append((raw_msg, seq, ner_tags, failed))

        return infer_result

    def do_infer(self, input_list: List[str]):
        std_input_list = [[i for i in text] for text in input_list]

        # do work, such like str2id and padding et al.
        preprocessed_msg = self.preprocess(std_input_list)

        # compose data structure and adding feature engineering data
        encoded_input_feature = self.encode_input_feature(preprocessed_msg)

        raw_predictions = self.predict_fn(encoded_input_feature)

        # extract predictions value from raw_predictions
        predictions = self.decode_output_feature(raw_predictions)

        # do work, such like id2str and so on
        ner_tag_sequence = self.postprocess(predictions)

        # decode ner tag and compose final result for return
        result = self.decode_ner_tag_sequence(
            ner_tag_sequence, std_input_list, input_list
        )

        return result
