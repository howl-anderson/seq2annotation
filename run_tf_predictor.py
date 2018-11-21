from tensorflow.contrib import predictor

from tokenizer_tools.tagset.NER.BILUO import BILUOSequenceEncoderDecoder

decoder = BILUOSequenceEncoderDecoder()

export_dir = 'results/saved_model/1542732555'

predict_fn = predictor.from_saved_model(export_dir)

text_msg = "王小明在北京的情话大学读书。"

predictions = predict_fn(
    {
        'words': [[i for i in text_msg]],
        'words_len': [len(text_msg)]
    }
)
print(predictions['tags'])

tags_seq = [i.decode() for i in predictions['tags'][0]]

word_list = decoder.decode_to_offset(tags_seq)
print(word_list)
