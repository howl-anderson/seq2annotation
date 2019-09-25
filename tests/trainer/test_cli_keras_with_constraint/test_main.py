import os

from seq2annotation.utils import remove_content_in_dir, create_dir_if_needed


def test_main():
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # set current working directory to current directory
    os.chdir(current_dir)

    # clean result dir first
    for target_dir in [
        os.path.join("./results", i)
        for i in ["h5_model", "model_dir", "saved_model", "summary_log_dir"]
    ]:
        create_dir_if_needed(target_dir)
        remove_content_in_dir(target_dir)

    import seq2annotation.trainer.cli_keras_with_constraint