from tensorflow.contrib import predictor

export_dir = 'results/saved_model/1540283398'

predict_fn = predictor.from_saved_model(export_dir)

text_msg = "王小明在北京的情话大学读书。"

predictions = predict_fn(
    {
        'words': [[i for i in text_msg]],
        'words_len': [len(text_msg)]
    }
)
print(predictions['tags'])
