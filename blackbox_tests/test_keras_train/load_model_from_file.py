import tensorflow as tf
import tf_crf_layer

from tf_crf_layer.metrics.crf_accuracy import crf_accuracy
from tf_crf_layer.metrics.sequence_span_accuracy import sequence_span_accuracy

model = tf.keras.models.load_model("./results/h5_model/model.h5", custom_objects={"crf_accuracy": crf_accuracy, "sequence_span_accuracy": sequence_span_accuracy})
model.summary()
