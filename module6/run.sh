#!/bin/bas

MAX_VALUE=1000

run_in_banner() {
	echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	echo "Running $@"
	echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	$@
	echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	echo
	echo
}
	

run_in_banner ./assignment 16 $MAX_VALUE 1 2 4 7 9 10

run_in_banner ./assignment 32 $MAX_VALUE 1 2 4 7 9 10
run_in_banner ./assignment 64 $MAX_VALUE 1 2 4 7 9 10
run_in_banner ./assignment 128 $MAX_VALUE 1 2 4 7 9 10
run_in_banner ./assignment 256 $MAX_VALUE 1 2 4 7 9 10
run_in_banner ./assignment 512 $MAX_VALUE 1 2 4 7 9 10

run_in_banner ./assignment 512 $MAX_VALUE 1
run_in_banner ./assignment 512 $MAX_VALUE 1 2
run_in_banner ./assignment 512 $MAX_VALUE 1 2 3
run_in_banner ./assignment 512 $MAX_VALUE 1 2 3 4 5 6
run_in_banner ./assignment 512 $MAX_VALUE 1 2 3 4 5 6 7 8 9
run_in_banner ./assignment 512 $MAX_VALUE 1 2 3 4 5 6 7 8 9 10 12 13 14
