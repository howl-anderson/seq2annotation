from typing import Union, Callable

from deliverable_model.metacontent import MetaContent
from deliverable_model.builder import (
    DeliverableModelBuilder,
    MetadataBuilder,
    ProcessorBuilder,
    ModelBuilder,
    RemoteModelBuilder,
)
from deliverable_model.builtin import LookupProcessor
from deliverable_model.builtin.processor import BILUOEncodeProcessor, PadProcessor
from seq2annotation_for_deliverable.main import (
    RemoteKerasConverterForRequest,
    RemoteKerasConverterForResponse,
)


def export_as_deliverable_model(
    output_dir,
    tensorflow_saved_model=None,
    converter_for_request: Union[None, Callable] = None,
    converter_for_response: Union[None, Callable] = None,
    keras_saved_model=None,
    keras_h5_model=None,
    meta_content_id="algorithmId-corpusId-configId-runId",
    vocabulary_lookup_table=None,
    tag_lookup_table=None,
    padding_parameter=None,
    addition_model_dependency=None,
    custom_object_dependency=None,
):
    # check parameters
    assert any(
        [tensorflow_saved_model, keras_saved_model, keras_h5_model]
    ), "one and only one of [tensorflow_saved_model, keras_saved_model, keras_h5_model] must be set up"
    assert (
        sum(
            int(bool(i))
            for i in [tensorflow_saved_model, keras_saved_model, keras_h5_model]
        )
        == 1
    ), "one and only one of [tensorflow_saved_model, keras_saved_model, keras_h5_model] must be set up"

    # default value
    addition_model_dependency = (
        [] if addition_model_dependency is None else addition_model_dependency
    )
    custom_object_dependency = (
        [] if custom_object_dependency is None else custom_object_dependency
    )

    # setup main object
    deliverable_model_builder = DeliverableModelBuilder(output_dir)

    # metadata builder
    metadata_builder = MetadataBuilder()
    meta_content = MetaContent(meta_content_id)
    metadata_builder.set_meta_content(meta_content)
    metadata_builder.save()

    # processor builder
    processor_builder = ProcessorBuilder()

    vocab_lookup_processor = LookupProcessor(vocabulary_lookup_table)
    vocab_lookup_processor_handle = processor_builder.add_processor(
        vocab_lookup_processor
    )

    tag_lookup_processor = LookupProcessor(tag_lookup_table)
    tag_lookup_processor_handle = processor_builder.add_processor(tag_lookup_processor)

    pad_processor = PadProcessor(padding_parameter=padding_parameter)
    pad_processor_handle = processor_builder.add_processor(pad_processor)

    decode_processor = BILUOEncodeProcessor()
    decoder_processor_handle = processor_builder.add_processor(decode_processor)

    ## pre process: encoder > vocab_lookup > pad
    processor_builder.add_preprocess(decoder_processor_handle)
    processor_builder.add_preprocess(vocab_lookup_processor_handle)
    processor_builder.add_preprocess(pad_processor_handle)

    ## post process: tag_lookup > encoder
    processor_builder.add_postprocess(tag_lookup_processor_handle)
    processor_builder.add_postprocess(decoder_processor_handle)

    processor_builder.save()

    # model builder
    model_builder = ModelBuilder()
    model_builder.append_dependency(addition_model_dependency)
    model_builder.set_custom_object_dependency(custom_object_dependency)
    model_builder.add_converter_for_request(converter_for_request)
    model_builder.add_converter_for_response(converter_for_response)
    model_builder.add_keras_saved_model(keras_saved_model)
    model_builder.save()

    # remote model builder
    remote_model_builder = RemoteModelBuilder("tf+grpc")
    remote_model_builder.add_converter_for_request(RemoteKerasConverterForRequest())
    remote_model_builder.add_converter_for_response(RemoteKerasConverterForResponse())
    remote_model_builder.save()

    # compose all the parts
    deliverable_model_builder.add_processor(processor_builder)
    deliverable_model_builder.add_metadata(metadata_builder)
    deliverable_model_builder.add_model(model_builder)
    deliverable_model_builder.add_remote_model(remote_model_builder)

    metadata = deliverable_model_builder.save()

    return metadata
