#!/bin/bash

chmod +x assignment.exe

(
./assignment.exe 50000 64
./assignment.exe 100000 64
./assignment.exe 200000 64
./assignment.exe 400000 64
./assignment.exe 800000 64
./assignment.exe 1600000 64
) 2> assignment_gpu_large_64_block_size.csv
