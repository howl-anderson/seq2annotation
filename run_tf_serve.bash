#!/usr/bin/env bash

docker run -p 8500:8500 -p 8501:8501 \
--mount type=bind,source=$(pwd)/results/saved_model,target=/models/seq2label \
-e MODEL_NAME=seq2label -t tensorflow/serving:nightly