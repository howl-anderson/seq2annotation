import setuptools
from setuptools import setup

from setuptools.command.install import install
from subprocess import getoutput


class PostInstall(install):
    pkg = 'git+https://github.com/guillaumegenthial/tf_metrics.git'

    def run(self):
        install.run(self)
        print(getoutput('pip install ' + self.pkg))


setup(
    name='seq2annotation',
    version='0.3.3',
    packages=setuptools.find_packages(),
    include_package_data=True,
    url='',
    license='MIT',
    author='Xiaoquan Kong',
    author_email='u1mail2me@gmail.com',
    description='seq2annotation',
    install_requires=['numpy', 'tensorflow', 'keras', 'tokenizer_tools', 'flask', 'flask-cors'],
    cmdclass={'install': PostInstall}
)
