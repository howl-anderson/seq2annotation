from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['DEBUG'] = True
CORS(app)

from tensorflow.contrib import predictor

export_dir = 'results/saved_model/1540283398'

predict_fn = predictor.from_saved_model(export_dir)


@app.route("/seq2annotation", methods=['GET'])
def single_tokenizer():
    text_msg = request.args.get('q')

    predictions = predict_fn(
        {
            'words': [[i for i in text_msg]],
            'words_len': [len(text_msg)]
        }
    )
    print(predictions['tags'])

    return jsonify(predictions['tags'])


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
