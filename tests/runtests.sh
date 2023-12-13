#!/bin/bash
pytest dataGenerator_tests/*
if [ $? -ne 0 ]; then
   exit $?
fi
pytest solver_tests/*
if [ $? -ne 0 ]; then
   exit $?
fi
pytest solver_tests_spinn/*
exit $?
