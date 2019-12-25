import os

import setuptools
from setuptools import setup

install_requires = [
    "numpy",
    "keras",
    "tokenizer_tools",
    "flask",
    "flask-cors",
    "ioflow",
    "tf-crf-layer",
    "tf-attention-layer",
    "tensorflow>=1.15.0,<2.0.0",
    "deliverable-model",
    "gunicorn",
    "micro_toolkit"
]


setup(
    name=os.getenv("_PKG_NAME", "seq2annotation"),  # _PKG_NAME will be used in Makefile for dev release
    version="0.9.1",
    packages=setuptools.find_packages(),
    include_package_data=True,
    url="https://github.com/howl-anderson/seq2annotation",
    license="Apache 2.0",
    author="Xiaoquan Kong",
    author_email="u1mail2me@gmail.com",
    description="seq2annotation",
    install_requires=install_requires,
)
