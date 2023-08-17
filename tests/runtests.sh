#!/bin/bash
pytest dataGenerator_tests/*
pytest solver_tests/*
if [ $# = 1 ]
then
    if [ $1 = '--with_gpu_tests' ]
    then
        pytest --gpu solver_tests_gpu/*
    fi
fi
echo "Tests done"
exit 0
