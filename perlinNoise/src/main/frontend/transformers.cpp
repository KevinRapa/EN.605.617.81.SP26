
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

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

