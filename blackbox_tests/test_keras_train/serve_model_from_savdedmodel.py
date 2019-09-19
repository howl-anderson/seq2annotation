from seq2annotation.server.tensorflow_keras_savedmodel_inference import Inference

inference = Inference('./results/saved_model', './results/h5_model/tag_lookup_table.json', './results/h5_model/vocabulary_lookup_table.json')
result = inference.infer("看一下上海的天气。")

print(result)