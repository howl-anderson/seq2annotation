import os

from seq2annotation.utils import remove_content_in_dir, create_dir_if_needed


def test_main():
    # clean result dir first
    for target_dir in [
        os.path.join("./results", i)
        for i in ["h5_model", "model_dir", "saved_model", "summary_log_dir"]
    ]:
        create_dir_if_needed(target_dir)
        remove_content_in_dir(target_dir)

    current_dir = os.path.dirname(os.path.realpath(__file__))

    config_file = os.path.join(current_dir, "./configure.yaml")

    os.environ["_DEFAULT_CONFIG_FILE"] = config_file

    # set current working directory to file directory
    os.chdir(current_dir)

    # TODO(howl-anderson): wrap up the train function into function call
    import seq2annotation.trainer.cli_keras
