#!/bin/bash
pytest dataGenerator_tests/*
status=$?
if [ $status -ne 0 ]; then
   exit $status
fi
pytest -s sharding_tests/*
status=$?
if [ $status -ne 0 ]; then
   exit $status
fi
pytest -s save_load_tests/*
status=$?
if [ $status -ne 0 ]; then
   exit $status
fi
pytest solver_tests/*
status=$?
if [ $status -ne 0 ]; then
   exit $status
fi
pytest solver_tests_spinn/*
status=$?
if [ $status -ne 0 ]; then
   exit $status
fi
pytest utils_tests/*
exit $?
