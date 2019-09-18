import setuptools
from setuptools import setup


def is_tensorflow_installed():
    """
    detect if tensorflow (no matter CPU or GPU based) installed

    :return: bool, True for tensorflow installed
    """
    import importlib

    try:
        importlib.import_module("tensorflow")
    except ModuleNotFoundError:
        return False

    return True


# without tensorflow by default
install_requires = [
    "numpy",
    "keras",
    "tokenizer_tools",
    "flask",
    "flask-cors",
    "ioflow",
]

if not is_tensorflow_installed():
    install_requires.append("tensorflow")  # Will install CPU based TensorFlow


setup(
    name="seq2annotation",
    version="0.6.3",
    packages=setuptools.find_packages(),
    include_package_data=True,
    url="https://github.com/howl-anderson/seq2annotation",
    license="Apache 2.0",
    author="Xiaoquan Kong",
    author_email="u1mail2me@gmail.com",
    description="seq2annotation",
    install_requires=install_requires,
)
