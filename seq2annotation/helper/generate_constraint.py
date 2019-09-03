import collections
from typing import List, Dict

from tokenizer_tools.tagset.offset.corpus import Corpus


def generate_constraint(
    corpus: Corpus,
    output_attr: str = None
) -> Dict[str, List[str]]:
    domain_list = collections.defaultdict(set)
    for item in corpus:
        if output_attr == "label":
            domain = item.label
        else:
            domain = item.extra_attr[output_attr]

        entity_list = []
        for span in item.span_set:
            entity = span.entity
            entity_list.append(entity)

        domain_list[domain].update(entity_list)

    domain_mapping = dict()

    # sort domain for stable output, more easy to compare and debug
    sorted_domain_list = sorted(domain_list.items(), key=lambda x: x[0])

    for k, v in sorted_domain_list:
        # sort entity for stable output, more easy to compare and debug
        sorted_entity_list = list(sorted(v))

        domain_mapping[k] = sorted_entity_list

    return domain_mapping
