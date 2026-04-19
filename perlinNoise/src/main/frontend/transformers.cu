
#include <transformers.h>

#include <curand.h>
#include <curand_kernel.h>
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

enum Threshold { LOW, MEDIUM, HIGH, VERY_HIGH };

__global__ void combineElevationAndHumdityLayersKernel(
	uchar3 *pixels,
	const float* elevation,
	const float* humidity,
	const float* details,
	float elevationMin,
	float elevationMax,
	float humidityMin,
	float humidityMax,
	float detailsMin,
	float detailsMax)
{
	unsigned globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

	enum Threshold elevationLevel;
	enum Threshold humidityLevel;

	float e = elevation[globalIdx];
	float h = humidity[globalIdx];
	float detailPixel = details[globalIdx];
	uchar3 pixelColor;

	curandState_t state;
	curand_init(1, globalIdx, 0, &state);

	// Set the elevation level
	if (e > elevationMax * 0.75) {
		elevationLevel = VERY_HIGH;
	} else if (e > elevationMax * 0.4) {
		elevationLevel = HIGH;
	} else if (e > elevationMax * 0.01) {
		elevationLevel = MEDIUM;
	} else {
		elevationLevel = LOW;
	}

	// Set the humidity level
	if ((h - humidityMin) > (humidityMax - humidityMin) * 0.6) {
		humidityLevel = HIGH;
	} else if ((h - humidityMin) > (humidityMax - humidityMin) * 0.5) {
		humidityLevel = MEDIUM;
	} else {
		humidityLevel = LOW;
	}

	// Set the pixel color based on elevation and humidity. Also populate trees/bushes with detail layer
	if (elevationLevel == VERY_HIGH) {
		// SNOWCAP
		float probabilityOfSnow;
		float draw = static_cast<float>(curand(&state) % 101);

		if (e > elevationMax * 0.90) {
			probabilityOfSnow = 100.0;
		} else if (e > elevationMax * 0.85) {
			probabilityOfSnow = 80.0;
		} else {
			probabilityOfSnow = 60.0;
		}

		if (draw < probabilityOfSnow) {
			pixelColor = make_uchar3(255, 255, 255);
		} else {
			pixelColor = make_uchar3(128, 128, 128);
		}
	} else if (elevationLevel == HIGH) {
		// MOUNTAIN
		pixelColor = make_uchar3(128, 128, 128);
	} else if (elevationLevel == LOW) {
		// RIVER
		pixelColor = make_uchar3(80, 235, 235);
	} else if (humidityLevel != LOW) {
		// FOREST
		float probabilityOfTree = ((detailPixel - detailsMin) / (detailsMax - detailsMin)) * 100.0;
		float draw = static_cast<float>(curand(&state) % 101);

		if (humidityLevel == MEDIUM) {
			probabilityOfTree /= 4.0;
		}

		if (draw < probabilityOfTree) {
			// There's a tree in this pixel
			pixelColor = make_uchar3(80, 185, 80);
		} else {
			pixelColor = make_uchar3(80, 255, 80);
		}
	} else {
		// PLAINS
		float probabilityOfBush = ((detailPixel - detailsMin) / (detailsMax - detailsMin)) * 100.0 * 0.1;
		float draw = static_cast<float>(curand(&state) % 101);

		if (draw < probabilityOfBush) {
			pixelColor = make_uchar3(140, 140, 80);
		} else {
			// There's a bush in this pixel
			pixelColor = make_uchar3(225, 225, 80);
		}
	}

	pixels[globalIdx] = pixelColor;
}

void combineElevationAndHumdityLayers(thrust::device_vector<uchar3> *pixels,
                                      const thrust::device_vector<float> *elevation,
                                      const thrust::device_vector<float> *humidity,
                                      const thrust::device_vector<float> *details,
                                      unsigned pixelWidth)
{
	unsigned threadsPerBlock = 512;
	unsigned numBlocks = (pixelWidth * pixelWidth) / threadsPerBlock;

	auto minMaxE = thrust::minmax_element(elevation->begin(), elevation->end());
	auto minMaxH = thrust::minmax_element(humidity->begin(), humidity->end());
	auto minMaxD = thrust::minmax_element(details->begin(), details->end());

	combineElevationAndHumdityLayersKernel<<<numBlocks, threadsPerBlock>>>(
	    thrust::raw_pointer_cast(pixels->data()),
	    thrust::raw_pointer_cast(elevation->data()),
	    thrust::raw_pointer_cast(humidity->data()),
	    thrust::raw_pointer_cast(details->data()),
	    *minMaxE.first, *minMaxE.second,
	    *minMaxH.first, *minMaxH.second,
	    *minMaxD.first, *minMaxD.second
	);
}
