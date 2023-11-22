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
if [ $? -ne 0 ]; then
   echo 'ERROR: pytest failed, exiting ...'
   exit $?
fi
echo "Tests done"
exit 0
