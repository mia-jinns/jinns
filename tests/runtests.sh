#!/bin/bash
pytest -s dataGenerator_tests/*
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
pytest -s solver_tests/*
status=$?
if [ $status -ne 0 ]; then
   exit $status
fi
pytest -s solver_tests_spinn/*
status=$?
if [ $status -ne 0 ]; then
   exit $status
fi
pytest -s utils_tests/*
exit $?
