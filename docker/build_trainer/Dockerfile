FROM ner_base:0.0.1

LABEL version="0.0.1-beta"

RUN mkdir /model
WORKDIR /model

ENV _DEFAULT_CONFIG_FILE=/data/configure.json
CMD ["python3", "-m", "seq2annotation.trainer.cli"]
