#!/usr/bin/env bash

docker run -p 5000:5000 -v /home/howl/workshop/seq2annotation/results/saved_model/1551737828:/model ner_base:0.0.1
