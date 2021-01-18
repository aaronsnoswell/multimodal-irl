#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="multimodal_irl",
    version="0.0.2",
    install_requires=[
        "gym >= 0.2.3",
        "numpy",
        "scipy",
        "sklearn",
        "numba",
        "networkx",
        "pandas >= 1.0.1",
        "matplotlib",
        "seaborn",
        "tqdm",
        "sacred",
        "pymongo",
        "cma",
        "mdp_extras @ git+https://github.com/aaronsnoswell/mdp-extras.git",
        "unimodal_irl @ git+https://github.com/aaronsnoswell/unimodal-irl.git",
    ],
    packages=find_packages(),
)
