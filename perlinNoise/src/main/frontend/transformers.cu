
#include <transformers.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <limits>

// This function is one of the simplest ways to map perlin noise values to displayable
// pixels. More interesting things can be done if I have the time
void convertNoiseToUchar3(uchar3 *pixels, const float *noise, unsigned pixelWidth)
{
	float minFloat = std::numeric_limits<float>::max();
	float maxFloat = std::numeric_limits<float>::lowest();
	float maxAlpha = std::numeric_limits<unsigned char>::max();

	// Get the minimum and maximum values
	for (int i = 0; i < pixelWidth * pixelWidth; i++) {
		if (noise[i] < minFloat) {
			minFloat = noise[i];
		}
		if (noise[i] > maxFloat) {
			maxFloat = noise[i];
		}
	}

	maxFloat -= minFloat;

	for (int i = 0; i < pixelWidth * pixelWidth; i++) {
		float percentOfMax = (noise[i] - minFloat) / maxFloat;

		unsigned char alpha = static_cast<unsigned char>(maxAlpha * percentOfMax);
		pixels[i] = make_uchar3(alpha, alpha, alpha);
	}
}

__global__ void convertBiomesTouchar3Kernel(uchar3 *pixels, const char *biomes, unsigned pixelWidth)
{
	unsigned globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

	switch (biomes[globalIdx]) {
	case MOUNTAIN:
		pixels[globalIdx] = make_uchar3(128, 128, 128);
		break;
	case FOREST:
		pixels[globalIdx] = make_uchar3(80, 255, 80);
		break;
	case PLAINS:
		pixels[globalIdx] = make_uchar3(255, 255, 80);
		break;
	case RIVER:
		pixels[globalIdx] = make_uchar3(80, 255, 255);
		break;
	default:
		pixels[globalIdx] = make_uchar3(0, 0, 0);
		break;
	}
}

void convertBiomesTouchar3(uchar3 *pixels, const char *biomes, unsigned pixelWidth)
{
	unsigned threadsPerBlock = 512;
	unsigned numBlocks = (pixelWidth * pixelWidth) / threadsPerBlock;

	convertBiomesTouchar3Kernel<<<numBlocks, threadsPerBlock>>>(pixels, biomes, pixelWidth);
}

enum ElevationThreshold { LOW, MEDIUM, HIGH };

__global__ void combineElevationAndHumdityLayersKernel(
	char *pixels,
	const float* elevation,
	const float* humidity,
	float elevationMin,
	float elevationMax,
	float humidityMin,
	float humidityMax)
{
	unsigned globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

	bool hIsHigh = humidity[globalIdx] > (humidityMin + humidityMax) / 2.0;
	enum ElevationThreshold eThresh;

	float e = elevation[globalIdx];
	char *out = pixels + globalIdx;

	if (e > elevationMax * 0.4) {
		eThresh = HIGH;
	} else if (e > elevationMax * 0.01) {
		eThresh = MEDIUM;
	} else {
		eThresh = LOW;
	}

	if (hIsHigh && eThresh == HIGH) {
		*out = MOUNTAIN;
	} else if (hIsHigh && eThresh == MEDIUM) {
		*out = FOREST;
	} else if (hIsHigh && eThresh == LOW) {
		*out = RIVER;
	} else if (!hIsHigh && eThresh == HIGH) {
		*out = MOUNTAIN;
	} else if (!hIsHigh && eThresh == MEDIUM) {
		*out = PLAINS;
	} else {
		*out = RIVER;
	}
}

void combineElevationAndHumdityLayers(thrust::device_vector<char> *pixels,
                                      const thrust::device_vector<float> *elevation,
                                      const thrust::device_vector<float> *humidity,
                                      unsigned pixelWidth)
{
	unsigned threadsPerBlock = 512;
	unsigned numBlocks = (pixelWidth * pixelWidth) / threadsPerBlock;

	auto minMaxE = thrust::minmax_element(elevation->begin(), elevation->end());
	auto minMaxH = thrust::minmax_element(humidity->begin(), humidity->end());

	combineElevationAndHumdityLayersKernel<<<numBlocks, threadsPerBlock>>>(
	    thrust::raw_pointer_cast(pixels->data()),
	    thrust::raw_pointer_cast(elevation->data()),
	    thrust::raw_pointer_cast(humidity->data()),
	    *minMaxE.first, *minMaxE.second,
	    *minMaxH.first, *minMaxH.second);
}
