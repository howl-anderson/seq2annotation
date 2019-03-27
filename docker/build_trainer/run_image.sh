#!/usr/bin/env bash

docker run -v `pwd`/../../:/model -v /../../configure.json:/data/configure.json ner_trainer:0.0.1
