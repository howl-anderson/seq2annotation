import sys

from flask import Flask, request, jsonify
from flask_cors import CORS
from tokenizer_tools.tagset.NER.BILUO import BILUOSequenceEncoderDecoder
from hanzi_char_lookup_feature.n_gram_lookup.ngrams_feature import generate_lookup_feature, load_data_set


decoder = BILUOSequenceEncoderDecoder()


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
# app.config['DEBUG'] = True
CORS(app)

from tensorflow.contrib import predictor

predict_fn = None

t = None


def load_t(t_data):
    global t
    t = load_data_set(
        {'person': [t_data]}
    )


def load_predict_fn(export_dir):
    global predict_fn
    predict_fn = predictor.from_saved_model(export_dir)

    return predict_fn


@app.route("/parse", methods=['GET'])
def single_tokenizer():
    text_msg = request.args.get('q')

    print(text_msg)

    # lookup_feature = generate_lookup_feature(t, text_msg, ['person'])
    lookup_feature = generate_lookup_feature(text_msg, 4, t, ['person'])

    input_feature = {
            'words': [[i for i in text_msg]],
            'words_len': [len(text_msg)],
            'lookup': [lookup_feature]
        }

    print(input_feature)

    predictions = predict_fn(input_feature)
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

    load_t(sys.argv[2])

    app.run(host='0.0.0.0', port=5000)
