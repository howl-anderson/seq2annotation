(cd ./build_base_image && docker rmi -f ner_base-cpu:0.0.1 && bash build_image.sh)
(cd ./build_trainer && docker rmi -f ner_trainer:0.0.1 && bash build_image.sh)
(cd ./build_server && docker rmi -f ner:0.0.1 && bash build_image.sh)