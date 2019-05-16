import sys

from flask import Flask, request, jsonify
from flask_cors import CORS

from seq2annotation.server.paddle_inference import Inference

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
# app.config['DEBUG'] = True
CORS(app)

server = None


def load_predict_fn(export_dir):
    global server
    server = Inference(export_dir)

    return server


@app.route("/parse", methods=['GET'])
def single_tokenizer():
    text_msg = request.args.get('q')

    print(text_msg)

    seq = server.infer(text_msg)

    response = {
        'text': text_msg,
        'spans': [{'start': i.start, 'end': i.end, 'type': i.entity} for i in
                  seq.span_set],
        'ents': list({i.entity.lower() for i in seq.span_set})
    }

    return jsonify(response)


if __name__ == "__main__":
    load_predict_fn(sys.argv[1])

    app.run(host='0.0.0.0', port=5000)
