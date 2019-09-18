from seq2annotation.server.tensorflow_keras_savedmodel_inference import Inference

inference = Inference('./results/saved_model')
result = inference.infer("看一下上海的天气。")

print("")