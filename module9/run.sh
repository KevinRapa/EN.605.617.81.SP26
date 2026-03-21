#!/bin/bash
# Generates three files of perlin noise. Files are comma-separated values
# Just from viewing them, you can sort of see a smooth gradient to the values

# These two generated files should be the same because the same seed is used
./main 0 > firstRun.txt
./main 0 > secondRun.txt

# This output should be different from the above two because a different seed is used
./main 483759 > differentSeed.txt
