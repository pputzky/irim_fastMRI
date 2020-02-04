#!/usr/bin/env bash

cwd=$(pwd)

pip install -r requirements.txt

pip install -r "${BASH_SOURCE%/*}/external/invertible_rim/requirements.txt"
cd  "${BASH_SOURCE%/*}/external/invertible_rim/"
python setup.py install
cd $cwd
