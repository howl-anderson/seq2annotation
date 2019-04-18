ARG BASE_IMAGE_NAME=ner_base-cpu
ARG BASE_IMAGE_TAG=0.0.1

FROM ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}

LABEL version="0.0.1-beta"

RUN mkdir /model

EXPOSE 5000
VOLUME /model

CMD ["python3", "-m", "seq2annotation.server.http", "/model"]
