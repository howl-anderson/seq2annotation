FROM ubuntu:16.04

LABEL version="0.0.1-beta"

# setup local mirror
COPY sources.list /tmp/sources.list
RUN cat /tmp/sources.list > /etc/apt/sources.list
RUN rm /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    python3-pip \
    python-dev \
    build-essential \
    git

RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple git+https://github.com/guillaumegenthial/tf_metrics.git
RUN pip3 install tokenizer_tools 
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple git+https://github.com/howl-anderson/seq2annotation.git
RUN mkdir /model

EXPOSE 5000
VOLUME /model

CMD ["python3", "-m", "seq2annotation.server.http", "/model"]
