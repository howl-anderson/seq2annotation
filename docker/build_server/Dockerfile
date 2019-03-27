FROM ner_base

LABEL version="0.0.1-beta"

RUN mkdir /model

EXPOSE 5000
VOLUME /model

CMD ["python3", "-m", "seq2annotation.server.http", "/model"]
