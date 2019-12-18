import os

import pytest

from seq2annotation.server.tensorflow_keras_h5_inference import Inference
from tokenizer_tools.tagset.offset.sequence import Sequence
from tokenizer_tools.tagset.offset.span import Span
from tokenizer_tools.tagset.offset.span_set import SpanSet


@pytest.mark.skip("tf crf layer don't work in tf 1.15")
def test_tensorflow_keras_h5_inference(datadir):
    workshop_dir = datadir

    model_file = os.path.join(workshop_dir, "./h5_model/model.h5")
    tag_lookup_file = os.path.join(workshop_dir, "./h5_model/tag_lookup_table.json")
    vocabulary_lookup_file = os.path.join(
        workshop_dir, "./h5_model/vocabulary_lookup_table.json"
    )

    inference = Inference(model_file, tag_lookup_file, vocabulary_lookup_file)
    result = inference.infer("看一下上海的天气。")

    expected = (
        "看一下上海的天气。",
        Sequence(
            text=["看", "一", "下", "上", "海", "的", "天", "气", "。"],
            span_set=SpanSet([Span(3, 5, "城市名", value=None, normal_value=None)]),
            id=None,
            label=None,
            extra_attr={},
        ),
        ["O", "O", "O", "B-城市名", "L-城市名", "O", "O", "O", "O"],
        False,
    )

    assert expected == result
