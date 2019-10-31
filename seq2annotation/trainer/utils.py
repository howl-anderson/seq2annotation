from deliverable_model import MetaContent
from deliverable_model.builder import (
    DeliverableModelBuilder,
    MetadataBuilder,
    ProcessorBuilder,
    ModelBuilder,
)
from deliverable_model.builtin import LookupProcessor
from deliverable_model.builtin.processor import BILUOEncodeProcessor


def export_as_deliverable_model(
    output_dir,
    tensorflow_saved_model=None,
    keras_saved_model=None,
    keras_h5_model=None,
    meta_content_id="algorithmId-corpusId-configId-runId",
    vocabulary_lookup_table=None,
    tag_lookup_table=None,
    padding_parameter=None
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

    decode_processor = BILUOEncodeProcessor()
    decoder_processor_handle = processor_builder.add_processor(decode_processor)

    lookup_processor = LookupProcessor(padding_parameter=padding_parameter)

    if vocabulary_lookup_table:
        lookup_processor.add_vocabulary_lookup_table(vocabulary_lookup_table)

    if tag_lookup_table:
        lookup_processor.add_tag_lookup_table(tag_lookup_table)

    if vocabulary_lookup_table or tag_lookup_table:
        lookup_processor_handle = processor_builder.add_processor(lookup_processor)

    # # pre process
    if vocabulary_lookup_table or tag_lookup_table:
        processor_builder.add_preprocess(lookup_processor_handle)
    processor_builder.add_preprocess(decoder_processor_handle)

    # # post process
    processor_builder.add_postprocess(decoder_processor_handle)
    if vocabulary_lookup_table or tag_lookup_table:
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
