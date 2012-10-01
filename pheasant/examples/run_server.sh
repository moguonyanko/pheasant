#!/bin/sh

PYTHONPATH=${PYTHONPATH}:`pwd`/../..
export PYTHONPATH

#echo $PYTHONPATH

PYPATH=`which python3`

${PYPATH} server.py

