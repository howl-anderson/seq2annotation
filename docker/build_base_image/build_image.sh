#!/usr/bin/env bash

# output docker image name with tag
output_image=${1:-ner_base-cpu:0.0.1}

# tensorflow type: CPU or GPU
arch=${2:-cpu}

docker build --no-cache --force-rm -f ./dockerfiles/${arch}.Dockerfile --tag ${output_image} .
