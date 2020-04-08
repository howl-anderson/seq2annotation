from seq2annotation.server.tensorflow_keras_h5_inference import Inference

inference = Inference('./results/h5_model/model.h5', './results/h5_model/tag_lookup_table.json', './results/h5_model/vocabulary_lookup_table.json')
result = inference.infer("看一下上海的天气。")

print(result)