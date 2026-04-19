/**
 * Procedural Perlin Noise Viewer
 * Spring 2026 - EN.605.617.81.SP26
 *
 * Derived from work by Anwar Haidar's Fractal Renderer found in the project gallery
 *
 * This program creates a pannable window which displays preocedurally generated Perlin Noise
 * Panning is accomplished by dragging the mouse. I saw that the Fractal Generator project
 * was doing something very similar, although I did not need zooming functionality, so I
 * used his methods using the GLFW OpenGL library and GLEW to create the window.
 */

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <transformers.h>
#include <perlin.h>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

class ProceduralPerlin
{
private:
	GLFWwindow* window;
	GLuint textureId;

	int windowWidth;
	int windowHeight;

	// Global coordinates determined by panning window
	double currentLocationX;
	double currentLocationY;
	double lastLocationX;
	double lastLocationY;

	bool mouseIsDragging;
	bool shouldRender;

	// Perlin Generator specific variables
	long perlinGeneratorSeed;
	unsigned octaves;
	uchar3 *canvas;

	thrust::device_vector<float> layerElevationD;
	thrust::device_vector<float> layerHumidityD;
	thrust::device_vector<float> layerDetailsD;
	thrust::device_vector<uchar3> canvasD;

	cudaStream_t stream1, stream2, stream3;

	void initializeOpenGL();

	// Callbacks for GLFW to call
	static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
	static void cursorPositionCallback(GLFWwindow* window, double x, double y);
	static void framebufferSizeCallback(GLFWwindow* window, int width, int height);

	// Handlers for mouse movement
	void checkIfDragging(int button, int action, int mods);
	void panWindow(double x, double y);

	// Generates one image of perlin noise and renders it on screen
	void renderField();
	void generateTerrain();

public:
	ProceduralPerlin(int width, int height, long seed, unsigned octaves);
	~ProceduralPerlin();

	void run();
};

ProceduralPerlin::ProceduralPerlin(int width, int height, long seed, unsigned octaves)
{
	this->windowWidth = width;
	this->windowHeight = height;
	this->currentLocationX = 0.0;
	this->currentLocationY = 0.0;
	this->mouseIsDragging = false;
	this->shouldRender = true;
	this->perlinGeneratorSeed = seed;
	this->octaves = octaves;

	initializeOpenGL();

	glfwSetMouseButtonCallback(this->window, this->mouseButtonCallback);
	glfwSetCursorPosCallback(this->window, this->cursorPositionCallback);
	glfwSetFramebufferSizeCallback(this->window, this->framebufferSizeCallback);
 
	perlinInit();

	// Initialize last just in case above functions fail
	this->canvas = new uchar3[this->windowWidth * this->windowHeight];

	this->layerElevationD = thrust::device_vector<float>(this->windowWidth * this->windowWidth);
	this->layerHumidityD = thrust::device_vector<float>(this->windowWidth * this->windowWidth);
	this->layerDetailsD = thrust::device_vector<float>(this->windowWidth * this->windowWidth);
	this->canvasD = thrust::device_vector<uchar3>(this->windowWidth * this->windowWidth);

	cudaStreamCreate(&this->stream1);
	cudaStreamCreate(&this->stream2);
	cudaStreamCreate(&this->stream3);
}

ProceduralPerlin::~ProceduralPerlin()
{
	if (this->textureId) {
		glDeleteTextures(1, &this->textureId);
	}

	if (this->window) {
		glfwDestroyWindow(this->window);
	}

	perlinDeinit();
	glfwTerminate();

	delete[] this->canvas;

	cudaStreamDestroy(this->stream1);
	cudaStreamDestroy(this->stream2);
	cudaStreamDestroy(this->stream3);
}

void ProceduralPerlin::initializeOpenGL()
{
	if (!glfwInit()) {
		fprintf(stderr, "Failed to initialize GLFW\n");
		exit(1);
	}

	// Request OpenGL 3.3 compatibility profile
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

	// Create window
	this->window = glfwCreateWindow(this->windowWidth, this->windowHeight, "Procedural Perlin Noise Viewer", NULL, NULL);

	if (!this->window) {
		fprintf(stderr, "Failed to create GLFW window\n");
		glfwTerminate();
		exit(1);
	}

	// Make OpenGL context current for this thread
	glfwMakeContextCurrent(this->window);

	// The GL callbacks must invoke methods on this object, so save a reference to `this` for gl to get later
	glfwSetWindowUserPointer(this->window, this);

	// Initialize GLEW
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		exit(1);
	}

	// Configure OpenGL state for 2D rendering
	glDisable(GL_DEPTH_TEST);
	glViewport(0, 0, this->windowWidth, this->windowHeight);

	// Initialize texture data
	glGenTextures(1, &this->textureId);
	glBindTexture(GL_TEXTURE_2D, this->textureId);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, this->windowWidth, this->windowHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

}

void ProceduralPerlin::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	ProceduralPerlin* viewer = reinterpret_cast<ProceduralPerlin*>(glfwGetWindowUserPointer(window));
	viewer->checkIfDragging(button, action, mods);
}

void ProceduralPerlin::cursorPositionCallback(GLFWwindow* window, double x, double y)
{
	ProceduralPerlin* viewer = reinterpret_cast<ProceduralPerlin*>(glfwGetWindowUserPointer(window));
	viewer->panWindow(x, y);
}

void ProceduralPerlin::framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void ProceduralPerlin::checkIfDragging(int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_PRESS) {
			this->mouseIsDragging = true;
			glfwGetCursorPos(this->window, &this->lastLocationX, &this->lastLocationY);
		} else if (action == GLFW_RELEASE) {
			this->mouseIsDragging = false;
		}
	}
}

void ProceduralPerlin::panWindow(double x, double y)
{
	if (!this->mouseIsDragging) {
		return;
	}

	// Calculate mouse movement delta in pixels
	double dx = x - this->lastLocationX;
	double dy = y - this->lastLocationY;

	// This modifier slows down the screen when dragging
	float scaleDown = 0.1;

	this->currentLocationX -= dx * scaleDown;
	this->currentLocationY += dy * scaleDown;

	this->lastLocationX = x;
	this->lastLocationY = y;
	this->shouldRender = true;
}

void ProceduralPerlin::generateTerrain()
{
	long locationX = static_cast<long>(this->currentLocationX);
	long locationY = static_cast<long>(this->currentLocationY);
	long seed = this->perlinGeneratorSeed;

	perlinDevice(&this->layerElevationD, seed, locationX, locationY, this->windowWidth, this->windowWidth, this->octaves, stream1);

	perlinDevice(&this->layerHumidityD, seed + 1, locationX, locationY, this->windowWidth, this->windowWidth, this->octaves, stream2);

	// Always use max octaves so that details look speckly
	perlinDevice(&this->layerDetailsD, seed + 2, locationX, locationY, this->windowWidth, this->windowWidth, 8, stream2);

	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);
	cudaStreamSynchronize(stream3);

	combineElevationAndHumdityLayers(&this->canvasD, &this->layerElevationD, &this->layerHumidityD, &this->layerDetailsD, this->windowWidth);

	thrust::copy(this->canvasD.begin(), this->canvasD.end(), this->canvas);
}

void ProceduralPerlin::renderField()
{
	if (this->shouldRender) {
		this->generateTerrain();

		glBindTexture(GL_TEXTURE_2D, this->textureId);

		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->windowWidth, this->windowHeight, GL_RGB, GL_UNSIGNED_BYTE, reinterpret_cast<unsigned char *>(this->canvas));

		this->shouldRender = false;
	}

	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, this->textureId);

	// Texture mapping stuff... TODO: Figure out what this does
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(-1.0f, -1.0f);  // Bottom-left
	glTexCoord2f(1.0f, 0.0f);
	glVertex2f( 1.0f, -1.0f);  // Bottom-right
	glTexCoord2f(1.0f, 1.0f);
	glVertex2f( 1.0f,  1.0f);  // Top-right
	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(-1.0f,  1.0f);  // Top-left
	glEnd();

	// Swap front/back buffers (double buffering)
	glfwSwapBuffers(this->window);
}

void ProceduralPerlin::run()
{
	glfwSetWindowTitle(this->window, "Minecreft");

	while (!glfwWindowShouldClose(this->window)) {
		glfwPollEvents();
		renderField();
	}
}

// 1024 happens to be the max that the perlin generator supports.
// Also, the Field wrapper around the perlin generator locks the noise to a square, so these are the same
static const unsigned WINDOW_WIDTH_MAX = 1024;
static const unsigned WINDOW_HEIGHT_MAX = 1024;

void printUsage() {
	printf("Usage: ./viewer [-h] [-o octaves] [-s seed]\n"
	       "\n"
	       "Options:\n"
	       "-h:         Print this\n"
	       "-o octaves: Specify number of octaves of noise to generate\n"
	       "-s seed:    Specify the global seed for the generator\n"
	);
}

int main(int argc, char** argv)
{
	// Defaults
	long seed = 0;
	unsigned octaves = 1;

	argv++;

	// Parse arguments if there are any
	while (argc > 1) {
		char *currentArg = argv[0];
		char *currentVal = argv[1];
		
		if (!strcmp("-o", currentArg)) {
			octaves = atoi(currentVal);
		} else if (!strcmp("-s", currentArg)) {
			seed = atoi(currentVal);
		} else if (!strcmp("-h", currentArg)) {
			printUsage();
			exit(EXIT_SUCCESS);
		} else {
			fprintf(stderr, "Unknown argument: %s\n", currentArg);
			printUsage();
			exit(EXIT_FAILURE);
		}

		argc -= 2;
		argv += 2;
	}

	cudaSetDevice(0);

	try {
		ProceduralPerlin viewer(WINDOW_WIDTH_MAX, WINDOW_HEIGHT_MAX, seed, octaves);
		viewer.run();
	} catch (const std::exception& e) {
		fprintf(stderr, "Error: %s\n", e.what());
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
