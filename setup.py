#!/usr/bin/env python

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here/'README.md').read_text(encoding='utf-8')

setup(
      name='uts',
      version='0.1',
      description='uts Algorithms',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Mário antunes',
      author_email='mariolpantunes@gmail.com',
      url='https://github.com/mariolpantunes/uts',
      packages=find_packages(),
      install_requires=['numpy>=1.21.5']
)
