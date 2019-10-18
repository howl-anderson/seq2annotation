FROM tensorflow/tensorflow:1.15.0-py3-jupyter

LABEL version="0.0.1-beta"

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y git

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

RUN pip3 install seq2annotation

# for fix a stupid bug cased by UCloud which always access /usr/bin/python as python bin
RUN ln -s /usr/bin/python3 /usr/bin/python

## Application related part start ##

RUN mkdir /app
RUN mkdir /data
WORKDIR /data

COPY builtin_configure.json /app/builtin_configure.json

EXPOSE 5000

# adjust to ucloud
ENV _DEFAULT_CONFIG_FILE=/data/configure.json
ENV _BUILTIN_CONFIG_FILE=/app/builtin_configure.json

CMD ["python3", "-m", "seq2annotation.trainer.cli"]