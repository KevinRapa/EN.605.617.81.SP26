#!/bin/bash

chmod +x assignment_cpu.exe

(
./assignment_cpu.exe 512 256
./assignment_cpu.exe 1024 256
./assignment_cpu.exe 2048 256
./assignment_cpu.exe 4096 256
./assignment_cpu.exe 8192 256
./assignment_cpu.exe 16384 256
) 2> assignment_cpu.csv
