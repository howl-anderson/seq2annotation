import os
import pathlib

import tensorflow as tf


def create_dir_if_needed(directory):
    # copied from https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python

    # if not os.path.exists(directory):
    if not tf.io.gfile.exists(directory):
        # os.makedirs(directory)
        tf.io.gfile.makedirs(directory)


def join_path(a, b):
    return os.path.join(a, str(pathlib.PurePosixPath(b)))
