import os
import runpy
import shutil
import sys
from pathlib import Path

from seq2annotation.helper.create_datadir import main as create_datadir
from tokenizer_tools.tagset.offset.corpus import Corpus


def main(
    corpus_file: str = None,
    config_file: str = None,
    workspace_dir: str = None,
    test_size=None,
    train_corpus: str = None,
    test_corpus: str = None,
) -> str:
    workspace = Path(workspace_dir)
    workspace.mkdir(exist_ok=True)

    data_dir = workspace / "data"
    data_dir.mkdir(exist_ok=True)
    results_dir = workspace / "results"
    results_dir.mkdir(exist_ok=True)

    # data
    create_datadir(corpus_file, data_dir, test_size, train_corpus, test_corpus)

    # copy config file
    shutil.copy(config_file, str(workspace))

    os.chdir(workspace)

    runpy.run_module("seq2annotation.trainer.cli", run_name="__main__")

    return workspace / "results" / "deliverable_model_dir"


if __name__ == "__main__":
    # TODO: better CLI interface for end users
    corpus_file = sys.argv[1]
    config_file = sys.argv[2]
    workspace_dir = sys.argv[3]

    main(corpus_file, config_file, workspace_dir)
