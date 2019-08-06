#!/usr/bin/env bash

data_dir=${1:-`pwd`/../blackbox_tests/test_docker}
config_file=${2:-`pwd`/../../blackbox_tests/test_ecarx_profile/configure.json}
builtin_config_file=${3:-`pwd`/../../blackbox_tests/test_ecarx_profile/builtin_configure.json}

docker run -p 9998:9998 -v ${data_dir}:/data -v ${config_file}:/data/configure.json -v ${builtin_config_file}:/data/builtin_configure.json ner_trainer:0.0.1
