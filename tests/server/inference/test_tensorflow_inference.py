import os

from seq2annotation.server.inference.tensorflow_inference import TensorFlowInference
from tokenizer_tools.tagset.offset.sequence import Sequence
from tokenizer_tools.tagset.offset.span import Span
from tokenizer_tools.tagset.offset.span_set import SpanSet


def test_tensorflow_inference(datadir):
    # TODO(howl-anderson): skip this test until model file oversize issue solved
    return

    workshop_dir = datadir

    model_dir = os.path.join(workshop_dir, "./saved_model")

    inference = TensorFlowInference(model_dir)
    result = inference.infer("看一下上海的天气。")

    print(result)

    expected = (
        "看一下上海的天气。",
        Sequence(
            text=["看", "一", "下", "上", "海", "的", "天", "气", "。"],
            span_set=SpanSet([Span(3, 5, "地点", value=None, normal_value=None)]),
            id=None,
            label=None,
            extra_attr={},
        ),
        ["O", "O", "O", "B-地点", "L-地点", "O", "O", "O", "O"],
        False,
    )

    assert expected == result
