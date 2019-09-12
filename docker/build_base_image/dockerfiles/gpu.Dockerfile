FROM tensorflow/tensorflow:1.14.0-gpu-py3

LABEL version="0.0.1-beta"

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

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

# RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple git+https://github.com/guillaumegenthial/tf_metrics.git
# RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple tokenizer_tools
# RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas
# RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple git+https://github.com/howl-anderson/ioflow.git
# RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple git+https://github.com/howl-anderson/seq2annotation.git

RUN pip3 install seq2annotation

# [weird behaviour] make sure tensorflow is GPU based
RUN pip3 uninstall -y tensorflow
RUN pip3 uninstall -y tensorflow-gpu
RUN pip3 install tensorflow-gpu~=1.14

# for fix a stupid bug cased by UCloud which always access /usr/bin/python as python bin
RUN ln -s /usr/bin/python3 /usr/bin/python