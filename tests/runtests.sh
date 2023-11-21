#!/bin/bash
pytest dataGenerator_tests/*
pytest solver_tests/*
pytest solver_tests_spinn/*
echo "Tests done"
exit 0
