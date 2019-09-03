#!/usr/bin/env python
from typing import List

from tokenizer_tools.conllz.tag_collector import collect_entity_to_file


def collect_tag_to_file(input_file_list: List[str], output_file: str):
    collect_entity_to_file(input_file_list, output_file)
