#!/bin/sh

PYTHONPATH=${PYTHONPATH}:`pwd`/../
export PYTHONPATH

#echo $PYTHONPATH

PYPATH=`which python3`

echo "Python version check..." 
${PYPATH} --version

echo "Start all test."

cd test/

${PYPATH} `pwd`/test_util.py
${PYPATH} `pwd`/test_logic.py
${PYPATH} `pwd`/test_settheory.py
${PYPATH} `pwd`/test_numtheory.py
${PYPATH} `pwd`/test_linear.py
${PYPATH} `pwd`/test_grptheory.py
${PYPATH} `pwd`/test_calculus.py
${PYPATH} `pwd`/test_geometry.py
${PYPATH} `pwd`/test_algorithm.py
${PYPATH} `pwd`/test_statistics.py

echo "All test finished."
