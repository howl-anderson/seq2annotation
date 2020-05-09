import sys
import shutil
from typing import Union

from tokenizer_tools.tagset.offset.corpus import Corpus
from pathlib import Path


def main(
    corpus_file: str = None,
    datadir: str = None,
    test_size: Union[int, float] = 0.1,
    train_corpus: str = None,
    test_corpus: str = None,
):
    result_dir = Path(datadir)

    corpus = None

    if corpus_file:
        corpus = Corpus.read_from_file(corpus_file)

        train, test = corpus.train_test_split(test_size=test_size)
        train.write_to_file(result_dir / "train.conllx")
        test.write_to_file(result_dir / "test.conllx")
    else:
        train = Corpus.read_from_file(train_corpus)
        test = Corpus.read_from_file(test_corpus)
        docs = [doc for doc in train] + [doc for doc in test]
        corpus = Corpus(docs)

        shutil.copy(train_corpus, datadir)
        shutil.copy(test_corpus, datadir)

    entities = {span.entity for doc in corpus for span in doc.span_set}

    with open(result_dir / "entity.txt", "wt") as fd:
        fd.write("\n".join(entities))


if __name__ == "__main__":
    # TODO: better CLI interface for end users
    corpus_file = sys.argv[1]
    datadir = sys.argv[2]

    main(corpus_file, datadir)
