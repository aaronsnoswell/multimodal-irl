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
        "explicit_env @ git+https://github.com/aaronsnoswell/explicit-env.git",
        "unimodal_irl @ git+https://github.com/aaronsnoswell/unimodal-irl.git",
        "puddle_world @ git+https://github.com/aaronsnoswell/puddle-world.git",
    ],
    packages=find_packages(),
)
