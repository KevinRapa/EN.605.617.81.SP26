
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

__device__ enum Threshold getElevationLevel(float elevation, float elevationMax)
{
	if (elevation > elevationMax * 0.75) {
		return VERY_HIGH;
	} else if (elevation > elevationMax * 0.4) {
		return HIGH;
	} else if (elevation > elevationMax * 0.01) {
		return MEDIUM;
	} else {
		return LOW;
	}
}

__device__ enum Threshold getHumidityLevel(float humidity, float humidityMin, float humidityMax)
{
	if ((humidity - humidityMin) > (humidityMax - humidityMin) * 0.6) {
		return HIGH;
	} else if ((humidity - humidityMin) > (humidityMax - humidityMin) * 0.5) {
		return MEDIUM;
	} else {
		return LOW;
	}
}

/**
 * Use a details pixel's percent of maximum as a probability that a feature exists in this pixel
 */
__device__ bool computeIfFeature(float detailPixel, float detailMax, float modifier, curandState_t *state)
{
	float probabilityOfFeature = (detailPixel / detailMax) * 100.0 * modifier;
	float randomDraw = static_cast<float>(curand(state) % 101);

	return randomDraw < probabilityOfFeature;
}

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

	const uchar3 BUSH_COLOR = make_uchar3(130, 130, 60);
	const uchar3 PLAINS_COLOR = make_uchar3(200, 200, 60);
	const uchar3 TREE_COLOR = make_uchar3(50, 185, 50);
	const uchar3 FOREST_FLOOR_COLOR = make_uchar3(50, 235, 50);
	const uchar3 SNOW_COLOR = make_uchar3(255, 255, 255);

	float e = elevation[globalIdx];
	uchar3 pixelColor;

	// Each detail pixel represents the probability of a feature existing in that pixel.
	// Anchor pixel value to zero to make math simpler
	float detailPixel = details[globalIdx] - detailsMin;
	detailsMax -= detailsMin;

	enum Threshold elevationLevel = getElevationLevel(e, elevationMax);
	enum Threshold humidityLevel = getHumidityLevel(humidity[globalIdx], humidityMin, humidityMax);

	curandState_t state;
	curand_init(1, globalIdx, 0, &state);

	// Set the pixel color based on elevation and humidity. Also populate trees/bushes with detail layer
	if (elevationLevel == VERY_HIGH) {
		// SNOWCAP
		pixelColor = SNOW_COLOR;
	} else if (elevationLevel == HIGH) {
		// MOUNTAIN. Modify brightness per elevation
		unsigned char color = 128 + static_cast<unsigned char>(70.0 * (((e / elevationMax) - 0.4) / 0.35));  // Magic. Sorry
		pixelColor = make_uchar3(color, color, color);
	} else if (elevationLevel == LOW) {
		// RIVER. Modify brightness per depth
		unsigned char color = 225 + static_cast<char>((e / elevationMax) / 0.01);  // Magic. This just looks fine.
		pixelColor = make_uchar3(80, color - 30, color);
	} else if (humidityLevel != LOW) {
		// FOREST
		float modifier = (humidityLevel == MEDIUM) ? 0.25 : 1.0;  // Make trees sparser near edge of biome.
		pixelColor = computeIfFeature(detailPixel, detailsMax, modifier, &state) ? TREE_COLOR : FOREST_FLOOR_COLOR;
	} else {
		// PLAINS
		pixelColor = computeIfFeature(detailPixel, detailsMax, 0.1, &state) ? BUSH_COLOR : PLAINS_COLOR;
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
