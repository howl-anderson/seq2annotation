import os


def create_dir_if_needed(directory):
    # copied from https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
    if not os.path.exists(directory):
        os.makedirs(directory)
