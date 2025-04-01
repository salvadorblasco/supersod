#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import sys
import os.path
from setuptools import setup, find_packages

if sys.platform == 'win32':
    import py2exe

setup(
    name = "supersod",
    version = "1.0.1",
    packages = find_packages('src/*.py'),
    package_dir={'':'src'},
    install_requires=['PyQt5', 'matplotlib', 'numpy', 'scipy'],
    # metadata to upload pypi
    author="Salvador Blasco",
    author_email = "salvador.blasco@gmail.com",
    description = "Computes results from SOD assays",
    license = "GPL",
    keywords = "SOD superoxide",
    classifiers=[
       "Development Status :: 5 - Production/Stable",
       "Topic :: Scientific/Engineering :: Chemistry",
       "Environment :: X11 Applications :: Qt",
       "Operating System :: OS Independent",
       "Programming Language :: Python :: 3.6",
       "Intended Audience :: Science/Research",
           "License :: OSI Approved :: GNU General Public License (GPL)"]
)

