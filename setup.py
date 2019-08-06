import setuptools
from setuptools import setup


setup(
    name='seq2annotation',
    version='0.5.0',
    packages=setuptools.find_packages(),
    include_package_data=True,
    url='https://github.com/howl-anderson/seq2annotation',
    license='Apache 2.0',
    author='Xiaoquan Kong',
    author_email='u1mail2me@gmail.com',
    description='seq2annotation',
    install_requires=['numpy', 'tensorflow', 'keras', 'tokenizer_tools', 'flask', 'flask-cors', 'ioflow']
)
