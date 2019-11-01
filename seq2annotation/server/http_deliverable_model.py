import sys
from typing import Union, List

from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf

from deliverable_model.builtin.processor.biluo_decode_processor import PredictResult
from deliverable_model.request import Request
from deliverable_model.serving import DeliverableModel

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
# app.config['DEBUG'] = True
CORS(app)

server = None  # type: DeliverableModel


def load_predict_fn(export_dir):
    global server

    server = DeliverableModel.load(export_dir)

    return server


def seq_to_http(predict: PredictResult):
    return {
        "text": predict.sequence.text,
        "spans": [
            {"start": i.start, "end": i.end, "type": i.entity}
            for i in predict.sequence.span_set
        ],
        "ents": list({i.entity.lower() for i in predict.sequence.span_set}),
    }


def compose_http_response(seq_or_seq_list: Union[PredictResult, List[PredictResult]]):
    if isinstance(seq_or_seq_list, list):
        result = [seq_to_http(i) for i in seq_or_seq_list]
    else:
        result = seq_to_http(seq_or_seq_list)

    return jsonify(result)


@app.route("/parse", methods=["GET"])
def single_tokenizer():
    text_msg = request.args.get("q")

    request_obj = Request([[i for i in text_msg]])

    response_obj = server.parse(request_obj)

    return compose_http_response(response_obj.data)


@app.route("/parse", methods=["POST"])
def batch_infer():
    text_msg = request.get_json()

    request_obj = Request([[j for j in i] for i in text_msg])

    response_obj = server.parse(request_obj)

    return compose_http_response(response_obj.data)


def simple_test():
    text_msg = "今天拉萨的天气。"

    request_obj = Request([[i for i in text_msg]])

    response_obj = server.parse(request_obj)

    print(response_obj.data)


if __name__ == "__main__":
    load_predict_fn(sys.argv[1])

    simple_test()

    app.run(host="0.0.0.0", port=5000)
