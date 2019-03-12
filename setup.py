from setuptools import setup

setup(
    name='seq2annotation',
    version='0.1.2',
    packages=['seq2annotation', 'seq2annotation.algorithms', 'seq2annotation.data_input', 'seq2annotation.server', 'seq2annotation.trainer'],
    url='',
    license='MIT',
    author='Xiaoquan Kong',
    author_email='u1mail2me@gmail.com',
    description='seq2annotation',
    install_requires=['numpy', 'tensorflow', 'tokenizer_tools', 'flask', 'flask-cors']
)
