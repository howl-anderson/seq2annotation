import os
import sys
from typing import Union, List

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pconf import Pconf

from deliverable_model.builtin.processor.biluo_decode_processor import PredictResult
from deliverable_model.serving import SimpleModelInference

current_dir = os.path.dirname(__file__)
http_root_dir = os.path.join(current_dir, "NLP_server_frontend")

app = Flask(__name__, static_url_path="", static_folder=http_root_dir)

app.config["JSON_AS_ASCII"] = False
# app.config['DEBUG'] = True
CORS(app)

server: SimpleModelInference = None


def load_predict_fn(export_dir, inference_batch_size=32):
    global server

    server = SimpleModelInference(export_dir, inference_batch_size)

    return server


def seq_to_http(predict: PredictResult):
    return {
        "text": "".join(predict.sequence.text),
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


@app.route("/", defaults={"path": "NER.html"})
def send_static(path):
    return send_from_directory(http_root_dir, path)


@app.route("/parse", methods=["GET"])
def single_tokenizer():
    text_msg: str = request.args.get("q")

    predict_result = list(server.parse([text_msg]))[0]

    return compose_http_response(predict_result)


@app.route("/parse", methods=["POST"])
def batch_infer():
    text_msg = request.get_json()

    predict_result_list = list(server.parse(text_msg))

    return compose_http_response(predict_result_list)


def warmup_test():
    text_msg = "今天拉萨的天气。"

    predict_result = list(server.parse([text_msg]))[0]

    print(predict_result)


if "gunicorn" in sys.modules:  # when called by gunicorn in production environment
    # disable output log to console
    import logging
    log = logging.getLogger("werkzeug")
    log.disabled = True

    Pconf.env(whitelist=["MODEL_PATH"])
    config = Pconf.get()

    deliverable_server = load_predict_fn(config["MODEL_PATH"], 128)

if __name__ == "__main__":
    deliverable_server = load_predict_fn(sys.argv[1], 128)

    warmup_test()

    threaded = True
    # set threaded to false because keras based models are not thread safety when prediction
    if deliverable_server.model_metadata["model"]["type"] in ["keras_h5_model", "keras_saved_model"]:
        threaded = False

    app.run(host="0.0.0.0", port=5000, threaded=threaded)
