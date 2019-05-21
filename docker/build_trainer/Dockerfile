ARG BASE_IMAGE_NAME=ner_base-cpu
ARG BASE_IMAGE_TAG=0.0.1

FROM ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}

LABEL version="0.0.1-beta"

RUN mkdir /data
WORKDIR /data

# adjust to ucloud
ENV _DEFAULT_CONFIG_FILE=/data/configure.json
CMD ["python3", "-m", "seq2annotation.trainer.cli"]
