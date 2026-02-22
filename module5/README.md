
This assignment implements an algorithm to multiply two matrices of variable size, and then add a third matrix to the result

Matrix sizes are specified on the command line.

The binary is called 'assignment' and can be run as follows:

    ./assignment matrix1rows matrix1cols matrix2rows matrix2cols [maxElementMagnitude]

    matrix1cols and matrix2rows must match. Limit is 32 rows/cols
    

Files:

Makefile -
    run 'make' to build with optimizations
    run 'make debug' to build without optimizations
    artifact will be named 'assignment'

run.sh -
    run after building to run 'assignment' with various input matrix sizes
    will output 'timings.txt' to showcase timings

assignment.cu -
util.cu -
util.h -
    source code

timings_example.txt -
    example of what 'timings.txt' will look like

typescript -
    console log of program usage
