all: test
	nvcc -o main --std=c++11 -Isrc/main/backend src/main/backend/*.cpp src/main/backend/*.cu

clean:
	rm -f perlinTest fieldTest main

test:
	nvcc -o perlinTest -rdc=true --std=c++11 -Isrc/main/backend src/test/backend/perlin.cu src/main/backend/perlin.cu src/main/backend/crc64.cpp
	nvcc -o fieldTest --std=c++11 -Isrc/main/backend src/test/backend/field.cpp src/main/backend/field.cpp src/main/backend/perlin.cu src/main/backend/crc64.cpp

