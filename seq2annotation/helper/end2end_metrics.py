import sys
import json

from tokenizer_tools.tagset.offset.corpus_metric import CorpusMetric
from tokenizer_tools.tagset.offset.corpus import Corpus


def main(gold: str, pred: str) -> dict:
    gold_corpus = Corpus.read_from_file(gold)
    pred_corpus = Corpus.read_from_file(pred)

    cm = CorpusMetric.create_from_corpus(gold_corpus, pred_corpus)

    return {
        "entity_f1_score": cm.entity_f1_score,
        "entity_accuracy_score": cm.entity_accuracy_score,
        "entity_precision_score": cm.entity_precision_score,
        "entity_recall_score": cm.entity_recall_score,
        "entity_classification_report": cm.entity_classification_report,
        "doc_entity_correctness": cm.doc_entity_correctness,
    }


if __name__ == "__main__":
    # TODO: better CLI interface for end users
    gold_corpus_file = sys.argv[1]
    predicted_corpus_file = sys.argv[2]

    metric = main(gold_corpus_file, predicted_corpus_file)

    print(json.dumps(metric))
