FROM ubuntu:16.04

LABEL version="0.0.1-beta"

RUN apt-get update && apt-get install -y \
    python-pip \
    python-dev \
    build-essential \
    git
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple git+https://github.com/howl-anderson/seq2annotation.git
RUN mkdir /model

EXPOSE 5000
VOLUME /model

CMD ['python3', '-m', 'seq2annotation.server.http', '/model']