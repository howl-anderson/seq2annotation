import os

import pytest

from seq2annotation.utils import remove_content_in_dir, create_dir_if_needed
from seq2annotation.trainer import cli_keras


# @pytest.mark.skip("tf crf don't work in tf 1.15")
def test_main(datadir):
    workshop_dir = datadir
    # TODO(howl-anderson): add a util to clean workshop
    # clean result dir first
    result_dir = os.path.join(workshop_dir, "./results")
    for target_dir in [
        os.path.join(result_dir, i)
        for i in [
            "h5_model",
            "model_dir",
            "saved_model",
            "summary_log_dir",
            "deliverable_model_dir",
        ]
    ]:
        create_dir_if_needed(target_dir)
        remove_content_in_dir(target_dir)

    config_file = os.path.join(workshop_dir, "./configure.yaml")

    os.environ["_DEFAULT_CONFIG_FILE"] = config_file

    # set current working directory to file directory
    os.chdir(workshop_dir)

    cli_keras.main()
