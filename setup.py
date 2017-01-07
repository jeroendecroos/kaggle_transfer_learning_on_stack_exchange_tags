#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

from setuptools import setup, find_packages

setup(
    name='kaggle_transfer_learning_on_stack_exchange_tags',
    version='0.1.0',
    author='Jeroen Decroos, Ilya Kashkarev, Matěj Korvas',
    description='Room 007 team\'s submissions to '
                'https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags',
    keywords="Kaggle, machine learning",
    url='https://github.com/jeroendecroos/kaggle_transfer_learning_on_stack_exchange_tags',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    # If there are any external dependencies, they can be specified as follows.
    # install_requires=['Mako>=1.0.0',
    #                   'PyYAML==3.11',
    #                   'requests==2.11.1',
    #                   ],
)
