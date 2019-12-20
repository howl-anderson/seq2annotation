import os

import tensorflow as tf
from tensorflow.python.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    BatchNormalization,
)
from tensorflow.python.keras.models import Sequential

from ioflow.configure import read_configure
from ioflow.corpus import get_corpus_processor
from seq2annotation.input import generate_tagset, Lookuper, index_table_from_file
from seq2annotation.trainer.utils import export_as_deliverable_model
from seq2annotation.utils import create_dir_if_needed, create_file_dir_if_needed
from tf_attention_layer.layers.global_attentioin_layer import GlobalAttentionLayer
from tf_crf_layer.layer import CRF
from tf_crf_layer.loss import ConditionalRandomFieldLoss
from tf_crf_layer.metrics import (
    SequenceCorrectness,
    SequenceSpanAccuracy
)
from tokenizer_tools.tagset.converter.offset_to_biluo import offset_to_biluo


# tf.enable_eager_execution()


def main():
    config = read_configure()

    corpus = get_corpus_processor(config)
    corpus.prepare()
    train_data_generator_func = corpus.get_generator_func(corpus.TRAIN)
    eval_data_generator_func = corpus.get_generator_func(corpus.EVAL)

    corpus_meta_data = corpus.get_meta_info()

    tags_data = generate_tagset(corpus_meta_data["tags"])

    train_data = list(train_data_generator_func())
    eval_data = list(eval_data_generator_func())

    tag_lookuper = Lookuper({v: i for i, v in enumerate(tags_data)})

    vocab_data_file = config.get("vocabulary_file")

    if not vocab_data_file:
        # load built in vocabulary file
        vocab_data_file = os.path.join(
            os.path.dirname(__file__), "../data/unicode_char_list.txt"
        )

    vocabulary_lookuper = index_table_from_file(vocab_data_file)

    def preprocss(data, maxlen):
        raw_x = []
        raw_y = []

        for offset_data in data:
            tags = offset_to_biluo(offset_data)
            words = offset_data.text

            tag_ids = [tag_lookuper.lookup(i) for i in tags]
            word_ids = [vocabulary_lookuper.lookup(i) for i in words]

            raw_x.append(word_ids)
            raw_y.append(tag_ids)

        if maxlen is None:
            maxlen = max(len(s) for s in raw_x)

        print(">>> maxlen: {}".format(maxlen))

        x = tf.keras.preprocessing.sequence.pad_sequences(
            raw_x, maxlen, padding="post"
        )  # right padding

        # lef padded with -1. Indeed, any integer works as it will be masked
        # y_pos = pad_sequences(y_pos, maxlen, value=-1)
        # y_chunk = pad_sequences(y_chunk, maxlen, value=-1)
        y = tf.keras.preprocessing.sequence.pad_sequences(
            raw_y, maxlen, value=0, padding="post"
        )

        return x, y

    MAX_SENTENCE_LEN = config.get("max_sentence_len", 25)

    train_x, train_y = preprocss(train_data, MAX_SENTENCE_LEN)
    test_x, test_y = preprocss(eval_data, MAX_SENTENCE_LEN)

    EPOCHS = config["epochs"]
    EMBED_DIM = config["embedding_dim"]
    USE_ATTENTION_LAYER = config.get("use_attention_layer", False)
    BiLSTM_STACK_CONFIG = config.get("bilstm_stack_config", [])
    BATCH_NORMALIZATION_AFTER_EMBEDDING_CONFIG = config.get(
        "use_batch_normalization_after_embedding", False
    )
    BATCH_NORMALIZATION_AFTER_BILSTM_CONFIG = config.get(
        "use_batch_normalization_after_bilstm", False
    )
    CRF_PARAMS = config.get("crf_params", {})

    vacab_size = vocabulary_lookuper.size()
    tag_size = tag_lookuper.size()

    model = Sequential()

    model.add(
        Embedding(vacab_size, EMBED_DIM, mask_zero=True, input_length=MAX_SENTENCE_LEN)
    )

    if BATCH_NORMALIZATION_AFTER_EMBEDDING_CONFIG:
        model.add(BatchNormalization())

    for bilstm_config in BiLSTM_STACK_CONFIG:
        model.add(Bidirectional(LSTM(return_sequences=True, **bilstm_config)))

    if BATCH_NORMALIZATION_AFTER_BILSTM_CONFIG:
        model.add(BatchNormalization())

    if USE_ATTENTION_LAYER:
        model.add(GlobalAttentionLayer())

    model.add(CRF(tag_size, name="crf", **CRF_PARAMS))

    # print model summary
    model.summary()

    callbacks_list = []

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=create_dir_if_needed(config["summary_log_dir"])
    )
    callbacks_list.append(tensorboard_callback)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(create_dir_if_needed(config["model_dir"]), "cp-{epoch:04d}.ckpt"),
        load_weights_on_restart=True,
        verbose=1,
    )
    callbacks_list.append(checkpoint_callback)

    metrics_list = []

    metrics_list.append(SequenceCorrectness())
    metrics_list.append(SequenceSpanAccuracy())

    loss_func = ConditionalRandomFieldLoss()
    # loss_func = crf_loss

    model.compile("adam", loss={"crf": loss_func}, metrics=metrics_list)
    model.fit(
        train_x,
        train_y,
        epochs=EPOCHS,
        validation_data=[test_x, test_y],
        callbacks=callbacks_list,
    )

    # Save the model
    model.save(create_file_dir_if_needed(config["h5_model_file"]))

    tf.keras.experimental.export_saved_model(
        model, create_dir_if_needed(config["saved_model_dir"])
    )

    export_as_deliverable_model(
        create_dir_if_needed(config["deliverable_model_dir"]),
        keras_saved_model=config["saved_model_dir"],
        vocabulary_lookup_table=vocabulary_lookuper,
        tag_lookup_table=tag_lookuper,
        padding_parameter={"maxlen": MAX_SENTENCE_LEN, "value": 0, "padding": "post"},
        addition_model_dependency=["tf-crf-layer"],
        custom_object_dependency=["tf_crf_layer"],
    )


if __name__ == "__main__":
    main()
