FROM tensorflow/tensorflow:1.13.1-gpu-py3

LABEL version="0.0.1-beta"

# for setup local mirror
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    apt-utils

# setup local mirror
COPY sources.list  /etc/apt/sources.list

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y git

RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple git+https://github.com/guillaumegenthial/tf_metrics.git
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple tokenizer_tools
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple git+https://github.com/howl-anderson/ioflow.git
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple git+https://github.com/howl-anderson/seq2annotation.git

# bugfix
COPY function_utils.py /usr/local/lib/python3.6/dist-packages/tensorflow/python/util/function_utils.py