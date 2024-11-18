#!/bin/bash
cd dataGenerator_tests
pytest -s
status=$?
if [ $status -ne 0 ]; then
   exit $status
fi
cd ../sharding_tests
pytest -s
status=$?
if [ $status -ne 0 ]; then
   exit $status
fi
cd ../save_load_tests
pytest -s
status=$?
if [ $status -ne 0 ]; then
   exit $status
fi
cd ../solver_tests
pytest -s --ignore=test_NSPipeFlow_x64.py
status=$?
if [ $status -ne 0 ]; then
   exit $status
fi
cd ../solver_tests_spinn
pytest -s
status=$?
if [ $status -ne 0 ]; then
   exit $status
fi
cd ../utils_tests
pytest -s
status=$?
if [ $status -ne 0 ]; then
   exit $status
fi
cd ../parameters_tests
pytest -s
if [ $status -ne 0 ]; then
   exit $status
fi
cd ../operator_tests
pytest -s
exit $?
