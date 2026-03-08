
all: 
	nvcc -o test --std=c++11 -Isrc/main/backend src/test/backend/field.cpp src/main/backend/field.cpp src/main/backend/crc64.cpp
	nvcc -o main --std=c++11 -Isrc/main/backend src/main/backend/*.cpp src/main/backend/*.cu
