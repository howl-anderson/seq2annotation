#!/usr/bin/env bash

data_dir=${1:-`pwd`/../../}
config_file=${2:-`pwd`/../../configure.json}

docker run -v ${data_dir}:/model -v ${config_file}:/data/configure.json ner_trainer:0.0.1
