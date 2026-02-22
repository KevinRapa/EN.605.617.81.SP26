#!/bin/bash

#Square sizes
echo "Square sizes:" >> timings.txt
./assignment 2 2 2 2 25  2>> timings.txt
./assignment 4 4 4 4 25  2>> timings.txt
./assignment 8 8 8 8 25  2>> timings.txt
./assignment 16 16 16 16 25  2>> timings.txt
./assignment 32 32 32 32 25  2>> timings.txt

echo >> timings.txt
echo "Weird sizes:" >> timings.txt

# Weird sizes
./assignment 7 7 7 7 25  2>> timings.txt
./assignment 5 5 5 1 25  2>> timings.txt
./assignment 32 24 24 3 25  2>> timings.txt
./assignment 32 32 32 1 25  2>> timings.txt

echo "Wrote to timings.txt"
