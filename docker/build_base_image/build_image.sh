#!/usr/bin/env bash

arch=${1:-cpu}
version=${2:-0.0.1}

docker build --force-rm -f ./dockerfiles/${arch}.Dockerfile --tag ner_base-${arch}:${version} .
