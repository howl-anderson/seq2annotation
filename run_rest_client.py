import json
import requests

from tokenizer_tools.tagset.BMES import BMESEncoderDecoder

bmes_decoder = BMESEncoderDecoder()


# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://192.168.8.155:8501/v1/models/seq2annotation:predict'


def main(input_str):
    # Compose a JSON Predict request (send JPEG image in base64).
    request_object = {
        "instances":
            [
                {
                    "words": [i for i in input_str],
                    "words_len": len(input_str)
                },
            ]
    }
    predict_request = json.dumps(request_object)

    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()
    prediction = response.json()['predictions'][0]
    print(prediction)

    tags = prediction['tags']

    word_tags_pair = list(zip(input_str, tags))
    word_list = bmes_decoder.decode_char_tag_pair(word_tags_pair)
    print(word_list)


if __name__ == '__main__':
    main("王小明在北京的清华大学读书。")
    # main("中国的首都是北京")
