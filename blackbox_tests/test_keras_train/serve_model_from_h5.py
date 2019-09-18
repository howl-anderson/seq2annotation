from seq2annotation.server.tensorflow_keras_h5_inference import Inference

inference = Inference('./results/h5_model/model.h5')
result = inference.infer("看一下上海的天气。")

print("")