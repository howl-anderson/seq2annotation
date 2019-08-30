FROM tensorflow/tensorflow:1.14.0-py3

LABEL version="0.0.1-beta"

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# for setup local mirror
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    apt-utils

# setup local mirror
COPY docker/build_base_image/sources.list  /etc/apt/sources.list

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y git

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

RUN pip3 install git+https://github.com/howl-anderson/ioflow.git

ADD ./ /temp/seq2annotation
RUN pip3 install /temp/seq2annotation

ENV HEALTH_CHECK_TRANSPONDER_PORT=9998

EXPOSE 9998

HEALTHCHECK --interval=5s --timeout=3s CMD curl --fail http://localhost:9998/ping || exit 1
