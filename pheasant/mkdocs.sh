#!/bin/sh

PYTHONPATH=${PYTHONPATH}:../
export PYTHONPATH

#Ocuur error for Sphinx adapt Python3.
sphinx-apidoc -F -f -o ../docs/ ../pheasant

cd ../docs/

make html

