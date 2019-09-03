import json

from seq2annotation.helper.generate_constraint import generate_constraint
from tokenizer_tools.tagset.offset.corpus import Corpus


def generate_constraint_to_file(
    input_file: str,
    output_file: str,
    output_attr: str = "label"
):
    corpus = Corpus.read_from_file(input_file)

    domain_mapping = generate_constraint(corpus, output_attr)

    with open(output_file, "wt") as fd:
        json.dump(domain_mapping, fd, indent=4, ensure_ascii=False)
