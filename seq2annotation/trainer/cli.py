from deliverable_model import MetaContent
from deliverable_model.builder import (
    DeliverableModelBuilder,
    MetadataBuilder,
    ProcessorBuilder,
    ModelBuilder,
)
from deliverable_model.builtin import LookupProcessor
from deliverable_model.utils import create_dir_if_needed
from ioflow.corpus import get_corpus_processor
from ioflow.eval_reporter import get_eval_reporter
from ioflow.task_status import get_task_status
from ioflow.model_saver import get_model_saver
from ioflow.configure import read_configure

from seq2annotation.input import build_input_func, generate_tagset, Lookuper
from seq2annotation.model import Model
from seq2annotation.health_check_transponder import (
    run_health_check_transponder_in_background,
)

# start health check
run_health_check_transponder_in_background()

raw_config = read_configure()
model = Model(raw_config)

config = model.get_default_config()
config.update(raw_config)

task_status = get_task_status(config)

task_status.send_status(task_status.START)

# read data according configure
# # report status: start to download corpus
task_status.send_status(task_status.START_DOWNLOAD_CORPUS)

corpus = get_corpus_processor(config)
corpus.prepare()
train_data_generator_func = corpus.get_generator_func(corpus.TRAIN)
eval_data_generator_func = corpus.get_generator_func(corpus.EVAL)

corpus_meta_data = corpus.get_meta_info()

config["tags_data"] = generate_tagset(corpus_meta_data["tags"])

# build model according configure


# # report status: start to process corpus
task_status.send_status(task_status.START_PROCESS_CORPUS)

# train and evaluate model
train_input_func = build_input_func(train_data_generator_func, config)
eval_input_func = (
    build_input_func(eval_data_generator_func, config)
    if eval_data_generator_func
    else None
)

# # report status: start to train
task_status.send_status(task_status.START_TRAIN)
task_status.send_progress(0)
task_status.send_progress(50)

evaluate_result, export_results, final_saved_model = model.train_and_eval_then_save(
    train_input_func, eval_input_func, config
)

task_status.send_progress(100)

# # report status: start to test
task_status.send_status(task_status.START_TEST)

from seq2annotation.server.tensorflow_inference import Inference
from seq2annotation.server.http import sequence_to_response

eval_reporter = get_eval_reporter(config)

# result_list = []
# inference = Inference(final_saved_model)
# for item in eval_data_generator_func():
#     text, result, _, _ = inference.infer("".join(item.text))
#     eval_reporter.record_x_and_y(item, sequence_to_response(text, result))
#     # result_list.append(result)

eval_reporter.submit()

# # report status: start to upload model
task_status.send_status(task_status.START_UPLOAD_MODEL)

model_saver = get_model_saver(config)
model_saver.save_model(final_saved_model)


def export_as_deliverable_model(
    output_dir,
    tensorflow_saved_model=None,
    keras_saved_model=None,
    keras_h5_model=None,
    meta_content_id="algorithmId-corpusId-configId-runId",
    vocabulary_lookup_table=None,
    tag_lookup_table=None,
):
    # check parameters
    assert any(
        [tensorflow_saved_model, keras_saved_model, keras_h5_model]
    ), "one and only one of [tensorflow_saved_model, keras_saved_model, keras_h5_model] must be set up"
    assert (
        sum(int(bool(i)) for i in [tensorflow_saved_model, keras_saved_model, keras_h5_model]) == 1
    ), "one and only one of [tensorflow_saved_model, keras_saved_model, keras_h5_model] must be set up"

    # setup main object
    deliverable_model_builder = DeliverableModelBuilder(output_dir)

    # metadata builder
    metadata_builder = MetadataBuilder()

    meta_content = MetaContent(meta_content_id)

    metadata_builder.set_meta_content(meta_content)

    metadata_builder.save()

    # processor builder
    processor_builder = ProcessorBuilder()

    lookup_processor = LookupProcessor()

    if vocabulary_lookup_table:
        lookup_processor.add_vocabulary_lookup_table(vocabulary_lookup_table)

    if tag_lookup_table:
        lookup_processor.add_tag_lookup_table(tag_lookup_table)

    if vocabulary_lookup_table or tag_lookup_table:
        lookup_processor_handle = processor_builder.add_processor(lookup_processor)
        processor_builder.add_preprocess(lookup_processor_handle)
        processor_builder.add_postprocess(lookup_processor_handle)

    processor_builder.save()

    # model builder
    model_builder = ModelBuilder()

    if tensorflow_saved_model:
        model_builder.add_tensorflow_saved_model(tensorflow_saved_model)
    elif keras_saved_model:
        model_builder.add_keras_saved_model(keras_saved_model)
    else:
        model_builder.add_keras_h5_model(keras_h5_model)

    model_builder.save()

    # compose all the parts
    deliverable_model_builder.add_processor(processor_builder)
    deliverable_model_builder.add_metadata(metadata_builder)
    deliverable_model_builder.add_model(model_builder)

    metadata = deliverable_model_builder.save()

    return metadata


export_as_deliverable_model(
    create_dir_if_needed(config["deliverable_model_dir"]), tensorflow_saved_model=final_saved_model
)

task_status.send_status(task_status.DONE)
