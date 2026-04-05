all: test viewer

viewer: clean
	nvcc -o viewer --std=c++11 -lglfw -lGL -lGLEW -lcublas -Isrc/main/backend -Isrc/main/frontend src/main/backend/field.cpp src/main/backend/crc64.cpp src/main/backend/*.cu src/main/frontend/*.cpp

clean:
	rm -f perlinTest fieldTest main transformersTest viewer

test: clean
	nvcc -o perlinTest -rdc=true --std=c++11 -Isrc/main/backend src/test/backend/perlin.cu src/main/backend/perlin.cu src/main/backend/crc64.cpp
	nvcc -o fieldTest --std=c++11 -Isrc/main/backend src/test/backend/field.cpp src/main/backend/field.cpp src/main/backend/perlin.cu src/main/backend/crc64.cpp
	nvcc -o transformersTest --std=c++11 -Isrc/main/frontend src/test/frontend/transformers.cpp src/main/frontend/transformers.cpp

