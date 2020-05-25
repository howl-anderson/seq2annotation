import sys

import deliverable_model.serving as dm
from tokenizer_tools.tagset.offset.corpus import Corpus
from micro_toolkit.sys import argv


def main(model_dir, gold_corpus_file, predicted_corpus_file, install_dependencies=True):
    gold_corpus = Corpus.read_from_file(gold_corpus_file)

    dm_model = dm.load(model_dir, install_dependencies=install_dependencies)

    docs = []
    docs_failed = []

    for gold_doc in gold_corpus:
        text = gold_doc.text
        id_ = gold_doc.id

        request = dm.make_request(query=[text])
        response = dm_model.inference(request)
        result = response.data[0]

        doc = result.sequence
        doc.id = id_

        if result.is_failed:
            doc.extra_attr["is_failed"] = True
            doc.extra_attr["exec_msg"] = result.exec_msg
            docs_failed.append(doc)
        else:
            docs.append(doc)

    predicted_corpus = Corpus(docs + docs_failed)
    predicted_corpus.write_to_file(predicted_corpus_file)

    print(len(docs_failed), len(docs))


if __name__ == "__main__":
    # TODO: better CLI interface for end users
    model_dir = sys.argv[1]
    gold_corpus_file = sys.argv[2]
    predicted_corpus_file = sys.argv[3]
    install_dependencies = argv.get_argv(4, True)

    main(model_dir, gold_corpus_file, predicted_corpus_file, bool(install_dependencies))
