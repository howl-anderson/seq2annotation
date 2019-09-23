import tensorflow as tf

from seq2annotation.server.inference.keras_inference_base import KerasInferenceBase
from tf_crf_layer.metrics import crf_accuracy, sequence_span_accuracy


class TensorFlowKerasSavedmodelInference(KerasInferenceBase):
    def instance_predict_fn(self):
        model = tf.keras.experimental.load_from_saved_model(
            self.model_path,
            custom_objects={
                "crf_accuracy": crf_accuracy,
                "sequence_span_accuracy": sequence_span_accuracy,
            },
        )

        return model.predict
