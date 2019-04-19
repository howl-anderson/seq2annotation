#!/usr/bin/env bash

base_image=${1:-ner_base-cpu}
tag=${2:-0.0.1}

docker build --force-rm --build-arg BASE_IMAGE_NAME=${base_image} --build-arg BASE_IMAGE_TAG=${tag} --tag ner_trainer:0.0.1 .
