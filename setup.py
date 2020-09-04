#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="multimodal_irl",
    version="0.0.1",
    install_requires=[
        "gym >= 0.2.3",
        "numpy",
        "scipy",
        "numba",
        "pandas >= 1.0.1",
        "matplotlib",
        "seaborn",
        "python-interface",
        "pytest",
        "explicit-env @ git+https://github.com/aaronsnoswell/explicit-env.git",
        "unimodal-irl @ git+https://github.com/aaronsnoswell/unimodal-irl.git",
        "puddle-world @ git+https://github.com/aaronsnoswell/puddle-world.git",
    ],
    packages=find_packages(),
)
