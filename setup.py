#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="pbsp",
    version="1.0.0",
    description="Predict the binding site of a protein from its sequence",
    author="Vineeth Chelur",
    author_email="crvineeth97@gmail.com",
    url="https://github.com/crvineeth97/protein-binding-site-prediction",
    # install_requires=["pytorch-lightning"],
    packages=find_packages(),
)
