ARG BASE_IMAGE=ner_base-cpu:0.0.1

FROM ${BASE_IMAGE}

LABEL version="0.0.1-beta"

RUN mkdir /model

EXPOSE 5000
VOLUME /model

CMD ["python3", "-m", "seq2annotation.server.http", "/model"]
