#!/bin/bash
pytest -s --cov=../jinns --ignore=solver_tests/test_NSPipeFlow_x64.py
status=$?
if [ $status -ne 0 ]; then
   exit $status
fi
exit $?
