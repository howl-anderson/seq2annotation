import sys

from flask import Flask, request, jsonify
from flask_cors import CORS
from tokenizer_tools.tagset.NER.BILUO import BILUOSequenceEncoderDecoder

decoder = BILUOSequenceEncoderDecoder()


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['DEBUG'] = True
CORS(app)

from tensorflow.contrib import predictor

predict_fn = None


def load_predict_fn(export_dir):
    global predict_fn
    predict_fn = predictor.from_saved_model(export_dir)

    return predict_fn


@app.route("/parse", methods=['GET'])
def single_tokenizer():
    text_msg = request.args.get('q')

    print(text_msg)

    predictions = predict_fn(
        {
            'words': [[i for i in text_msg]],
            'words_len': [len(text_msg)]
        }
    )
    print(predictions['tags'])

    tags_seq = [i.decode() for i in predictions['tags'][0]]

    offset_list = decoder.decode_to_offset(tags_seq)
    print(offset_list)

    response = {
        'text': text_msg,
        'spans': [{'start': i[0], 'end': i[1], 'type': i[2]} for i in offset_list],
        'ents': list({i[2].lower() for i in offset_list})
    }

    return jsonify(response)


if __name__ == "__main__":
    load_predict_fn(sys.argv[1])

    app.run(host='0.0.0.0', port=5000)
