import os
import sys

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from pconf import Pconf

import deliverable_model.serving as dm

http_root_dir = os.path.join(os.path.dirname(__file__), "NLP_server_frontend")

app = Flask(__name__, static_url_path="", static_folder=http_root_dir)

app.config["JSON_AS_ASCII"] = False
# app.config['DEBUG'] = True
CORS(app)

model = None


class Model:
    def __init__(self, model_dir):
        self.dm_model = dm.load(model_dir)

    def parse(self, query, single_query=False):
        request = dm.make_request(query=query)
        response = self.dm_model.inference(request)

        result = response.data

        flask_response = self._compose_http_response(result)

        return flask_response

    def _compose_http_response(self, seq_list):
        return [self._seq_to_http(i) for i in seq_list]

    def _seq_to_http(self, predict):
        return {
            "text": "".join(predict.sequence.text),
            "spans": [
                {"start": i.start, "end": i.end, "type": i.entity}
                for i in predict.sequence.span_set
            ],
            "ents": list({i.entity.lower() for i in predict.sequence.span_set}),
            "is_failed": predict.is_failed,
            "exec_msg": predict.exec_msg,
        }


def load_model(export_dir):
    global model

    model = Model(export_dir)

    return model


@app.route("/", defaults={"path": "NER.html"})
def dashboard(path):
    return send_from_directory(http_root_dir, path)


@app.route("/parse", methods=["GET"])
def single_infer():
    text_msg: str = request.args.get("q")

    result = model.parse([text_msg])

    return jsonify(result[0])


@app.route("/parse", methods=["POST"])
def batch_infer():
    text_msg = request.get_json()

    result = model.parse(text_msg)

    return jsonify(result)


def warmup_test():
    with app.app_context():
        text_msg = "今天拉萨的天气。"

        predict_result = model.parse([text_msg])

        print(predict_result)


if "gunicorn" in sys.modules:  # when called by gunicorn in production environment
    # disable output log to console
    import logging

    log = logging.getLogger("werkzeug")
    log.disabled = True

    Pconf.env(whitelist=["MODEL_PATH"])
    config = Pconf.get()

    deliverable_server = load_model(config["MODEL_PATH"])

if __name__ == "__main__":
    deliverable_server = load_model(sys.argv[1])

    warmup_test()

    threaded = True
    # set threaded to false because keras based models are not thread safety when prediction
    # if deliverable_server.model_metadata["model"]["type"] in [
    #    "keras_h5_model",
    #    "keras_saved_model",
    # ]:
    #    threaded = False
    threaded = False

    app.run(host="0.0.0.0", port=5000, threaded=threaded)
