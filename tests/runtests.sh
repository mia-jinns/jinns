#!/bin/bash
pytest test_DataGeneratorODE.py test_CubicMeshPDEStatio.py test_CubicMeshPDENonStatio.py
pytest test_GLV_x32.py test_GLV_x64.py
if [ $# = 1 ]
then
    if [ $1 = '--with_gpu_tests' ]
    then
        pytest --gpu test_GLV_x32_gpu.py test_GLV_x64_gpu.py
    fi
fi
echo "Tests done"
exit 0
