#!/usr/bin/env bash

arch=${1:-cpu}
version=${2:-0.0.1}
prefix_tag=${3:-}  # docker tag prefix used for docker push

base_image=${prefix_tag}ner_base-${arch}:${version}
(cd ./build_base_image && docker rmi -f ${base_image} && bash build_image.sh ${base_image} ${arch})

trainer_image=${prefix_tag}ner_trainer-${arch}:${version}
test $? -eq 0 && (cd ./build_trainer && docker rmi -f ${trainer_image} && bash build_image.sh ${base_image} ${trainer_image})

server_image=${prefix_tag}ner-${arch}:${version}
test $? -eq 0 && (cd ./build_server && docker rmi -f ${server_image} && bash build_image.sh ${base_image} ${server_image})

if test $? -eq 0; then
    echo "Success!"
else
    echo "Failed!"
fi