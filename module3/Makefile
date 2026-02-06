all: assignment.c assignment.cu
	g++ -O0 assignment.c -o assignment_cpu.exe
	nvcc -O0 assignment.cu -o assignment.exe

cpu: assignment.c
	g++ -O0 assignment.c -o assignment_cpu.exe
	
gpu: assignment.cu
	nvcc -O0 assignment.cu -o assignment.exe
