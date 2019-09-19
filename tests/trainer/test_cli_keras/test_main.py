import os


def test_main():
    current_dir = os.path.dirname(os.path.realpath(__file__))

    config_file = os.path.join(current_dir, "./configure.yaml")

    os.environ["_DEFAULT_CONFIG_FILE"] = config_file

    import seq2annotation.trainer.cli_keras
