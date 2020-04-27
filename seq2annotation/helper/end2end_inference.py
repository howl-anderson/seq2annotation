import sys

import deliverable_model.serving as dm
from tokenizer_tools.tagset.offset.corpus import Corpus
from tokenizer_tools.tagset.offset.document import Document

# TODO: better CLI interface for end users
model_dir = sys.argv[1]
gold_corpus_file = sys.argv[2]
predicted_corpus_file = sys.argv[3]

gold_corpus = Corpus.read_from_file(gold_corpus_file)

dm_model = dm.load(model_dir)

docs = []
docs_failed = []

for gold_doc in gold_corpus:
    text = gold_doc.text
    id_ = gold_doc.id

    request = dm.make_request(query=[text])
    response = dm_model.inference(request)
    result = response.data[0]

    if result.is_failed:
        # TODO: Once new BILUO decoder released, this path can be removed
        doc = Document(text=text)
        doc.id = id_
        doc.extra_attr["is_failed"] = True
        doc.extra_attr["exec_msg"] = result.exec_msg
        docs_failed.append(doc)
    else:
        doc = result.sequence
        doc.id = id_
        docs.append(doc )

Corpus(docs + docs_failed).write_to_file(predicted_corpus_file)

print(len(docs_failed), len(docs))
