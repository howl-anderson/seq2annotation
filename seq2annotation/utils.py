import os
import pathlib
from typing import Text, Any, Type


def create_dir_if_needed(directory):
    # copied from https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python

    # if not os.path.exists(directory):
    if not os.path.exists(directory):
        # os.makedirs(directory)
        os.makedirs(directory)


def join_path(a, b):
    return os.path.join(a, str(pathlib.PurePosixPath(b)))


def class_from_module_path(module_path: Text) -> Type[Any]:
    # copied from rasa_nlu (https://github.com/RasaHQ/rasa) @ rasa_nlu/utils/__init__.py
    """Given the module name and path of a class, tries to retrieve the class.

    The loaded class can be used to instantiate new objects. """
    import importlib

    # load the module, will raise ImportError if module cannot be loaded
    if "." in module_path:
        module_name, _, class_name = module_path.rpartition('.')
        m = importlib.import_module(module_name)
        # get the class, will raise AttributeError if class cannot be found
        return getattr(m, class_name)
    else:
        return globals()[module_path]
    

def load_hook(hook_config):
    hook_instances = []
    for i in hook_config:
        class_ = class_from_module_path(i['class'])
        hook_instances.append(class_(**i.get('params', {})))
        
    return hook_instances
