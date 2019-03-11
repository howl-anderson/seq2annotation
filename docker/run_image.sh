#!/usr/bin/env bash

docker rum --publish 5000:5000 --volume /model:/Users/howl/PyCharmProjects/seq2annotation_ner_on_ecarx/results/saved_model/1549978457 --name ner ner