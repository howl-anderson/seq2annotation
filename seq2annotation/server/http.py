import os
import sys

from flask import Flask, request, jsonify
from flask_cors import CORS
from tokenizer_tools.tagset.NER.BILUO import BILUOSequenceEncoderDecoder
from tokenizer_tools.tagset.offset.sequence import Sequence

decoder = BILUOSequenceEncoderDecoder()


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
# app.config['DEBUG'] = True
CORS(app)

from tensorflow.contrib import predictor

server = None


def load_predict_fn(export_dir):
    global server
    server = Server(export_dir)

    return server


class IndexToStringTable(object):
    def __init__(self, lookup_file):
        self.dict = {}

        with open(lookup_file) as fd:
            for index, string in enumerate(fd):
                self.dict[index] = string.strip()

    def lookup(self, index):
        return self.dict[index]


class IndexTable(object):
    def __init__(self, lookup_file):
        self.dict = {}

        with open(lookup_file) as fd:
            for index, string in enumerate(fd):
                string = string.strip()
                self.dict[string] = index

    def lookup(self, string):
        return self.dict[string]


class Server(object):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.predict_fn = predictor.from_saved_model(model_dir)

        self.char_to_index_table = IndexTable(os.path.join(model_dir, 'assets.extra/vocab.txt'))
        self.index_to_tag_table = IndexToStringTable(os.path.join(model_dir, 'assets.extra/tags.txt'))

    def serve(self, input_text, raise_exception=False):
        # input_text = list(map(self.char_to_index_table.lookup, input_text))

        input_feature = {
            'words': [[i for i in input_text]],
            'words_len': [len(input_text)],
        }

        print(input_feature)

        predictions = self.predict_fn(input_feature)
        tags = predictions['tags'][0]
        # print(predictions['tags'])

        # decode Unicode
        tags_seq = [i.decode() for i in tags]

        print(tags_seq)

        # BILUO to offset
        failed = False
        try:
            seq = decoder.to_offset(tags_seq, input_text)
        except:
            if not raise_exception:
                # invalid tag sequence will raise exception
                # so return a empty result
                seq = Sequence(input_text)
                failed = True
            else:
                raise
        # print(seq)

        return seq, tags_seq, failed


@app.route("/parse", methods=['GET'])
def single_tokenizer():
    text_msg = request.args.get('q')

    print(text_msg)

    seq, tags, failed = server.serve(text_msg)

    print(tags)
    # print(seq)

    response = {
        'text': text_msg,
        'spans': [{'start': i.start, 'end': i.end, 'type': i.entity} for i in seq.span_set],
        'ents': list({i.entity.lower() for i in seq.span_set})
    }

    return jsonify(response)


if __name__ == "__main__":
    load_predict_fn(sys.argv[1])

    app.run(host='0.0.0.0', port=5000)
