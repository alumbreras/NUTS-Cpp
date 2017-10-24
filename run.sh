#!/bin/bash

g++ -g --std=c++11 nuts.cpp -o nuts -DARMA_DONT_USE_WRAPPER -lblas -llapack
valgrind --tool=callgrind ./nuts
kcachegrind `ls -t callgrind.out.* | head -1`
