FROM ner_base

LABEL version="0.0.1-beta"

## Application related part start ##

RUN mkdir /app
RUN mkdir /data
WORKDIR /data

COPY builtin_configure.json /app/builtin_configure.json
COPY entry.bash /app/entry.bash

EXPOSE 5000

# adjust to ucloud
ENV _BUILTIN_CONFIG_FILE=/app/builtin_configure.json

CMD ["bash", "/app/entry.bash"]
