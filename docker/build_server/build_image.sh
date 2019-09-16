#!/usr/bin/env bash

base_image=${1:-ner_base-cpu:0.0.1}
output_image=${2:-ner_cpu:0.0.1}

docker build --no-cache --force-rm --build-arg BASE_IMAGE=${base_image} --tag ${output_image} .
