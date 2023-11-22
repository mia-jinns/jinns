#!/bin/bash
pytest dataGenerator_tests/*
if [ $? -ne 0 ]; then
   echo 'ERROR: pytest failed, exiting ...'
   exit $?
fi
pytest solver_tests/*
if [ $? -ne 0 ]; then
   echo 'ERROR: pytest failed, exiting ...'
   exit $?
fi
pytest solver_tests_spinn/*
echo "Tests done"
exit $?
