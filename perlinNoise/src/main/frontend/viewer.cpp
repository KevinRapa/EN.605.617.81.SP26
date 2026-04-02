#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <transformers.h>
#include <field.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

class ProceduralPerlin
{
private:
	GLFWwindow* window;
	GLuint textureId;
	GLuint pbo;
	struct cudaGraphicsResource* cudaPboResource;

	int windowWidth;
	int windowHeight;
	double currentLocationX;
	double currentLocationY;
	double lastLocationX;
	double lastLocationY;
	bool mouseIsDragging;
	bool shouldRender;
	long perlinGeneratorSeed;
	uchar3* canvas;

	void initializeOpenGL();

	// Callbacks for GLFW to call
	static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
	static void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos);
	static void framebufferSizeCallback(GLFWwindow* window, int width, int height);

	// Handlers for mouse movement
	void checkIfDragging(int button, int action, int mods);
	void panWindow(double xpos, double ypos);

	// Generates one image of perlin noise and renders it on screen
	void renderField();

public:
	ProceduralPerlin(int width, int height, long seed);
	~ProceduralPerlin();

	void run();
};

ProceduralPerlin::ProceduralPerlin(int width, int height, long seed)
{
	this->windowWidth = width;
	this->windowHeight = height;
	this->currentLocationX = 0.0;
	this->currentLocationY = 0.0;
	this->mouseIsDragging = false;
	this->shouldRender = true;
	this->perlinGeneratorSeed = seed;
	this->canvas = new uchar3[this->windowWidth * this->windowHeight];

	initializeOpenGL();

	cudaGraphicsGLRegisterBuffer(&this->cudaPboResource, this->pbo, cudaGraphicsMapFlagsWriteDiscard);

	glfwSetMouseButtonCallback(this->window, this->mouseButtonCallback);
	glfwSetCursorPosCallback(this->window, this->cursorPositionCallback);
	glfwSetFramebufferSizeCallback(this->window, this->framebufferSizeCallback);
}

ProceduralPerlin::~ProceduralPerlin()
{
	if (this->cudaPboResource) {
		cudaGraphicsUnregisterResource(this->cudaPboResource);
	}
	if (this->pbo) {
		glDeleteBuffers(1, &this->pbo);
	}
	if (this->textureId) {
		glDeleteTextures(1, &this->textureId);
	}
	if (this->window) {
		glfwDestroyWindow(this->window);
	}
	glfwTerminate();

	delete[] this->canvas;
}

void ProceduralPerlin::initializeOpenGL()
{
	// Initialize GLFW library
	if (!glfwInit()) {
		fprintf(stderr, "Failed to initialize GLFW!\n");
		exit(1);
	}

	// Request OpenGL 3.3 compatibility profile
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

	// Create window
	this->window = glfwCreateWindow(this->windowWidth, this->windowHeight, "CUDA Fractal Explorer", NULL, NULL);
	if (!this->window) {
		fprintf(stderr, "Failed to create GLFW window!\n");
		glfwTerminate();
		exit(1);
	}

	// Make OpenGL context current for this thread
	glfwMakeContextCurrent(this->window);

	// DISABLE V-SYNC to show true GPU performance (not limited by monitor refresh)
	glfwSwapInterval(0);  // 0 = V-Sync OFF, 1 = V-Sync ON

	glfwSetWindowUserPointer(this->window, this);

	// Initialize GLEW to load OpenGL extensions
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW!\n");
		exit(1);
	}

	// Configure OpenGL state
	glDisable(GL_DEPTH_TEST);  // 2D rendering
	glViewport(0, 0, this->windowWidth, this->windowHeight);

	// Create texture to display fractal image
	glGenTextures(1, &this->textureId);
	glBindTexture(GL_TEXTURE_2D, this->textureId);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, this->windowWidth, this->windowHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

	glGenBuffers(1, &this->pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, this->pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, this->windowWidth * this->windowHeight * sizeof(uchar3), nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void ProceduralPerlin::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	ProceduralPerlin* viewer = static_cast<ProceduralPerlin*>(glfwGetWindowUserPointer(window));
	viewer->checkIfDragging(button, action, mods);
}

void ProceduralPerlin::cursorPositionCallback(GLFWwindow* window, double xpos, double ypos)
{
	ProceduralPerlin* viewer = static_cast<ProceduralPerlin*>(glfwGetWindowUserPointer(window));
	viewer->panWindow(xpos, ypos);
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

void ProceduralPerlin::panWindow(double xpos, double ypos)
{
	if (!this->mouseIsDragging) {
		return;
	}

	// Calculate mouse movement delta in pixels
	double dx = xpos - this->lastLocationX;
	double dy = ypos - this->lastLocationY;

	// This modifier slows down the screen when dragging
	float scaleDown = 0.1;

	this->currentLocationX -= dx * scaleDown;
	this->currentLocationY += dy * scaleDown;

	this->lastLocationX = xpos;
	this->lastLocationY = ypos;
	this->shouldRender = true;
}

void ProceduralPerlin::renderField()
{
	if (this->shouldRender) {
		static float *field = nullptr;
		if (!field) {
			field = fieldAlloc(this->windowWidth);
		}

		createField(field, this->perlinGeneratorSeed, this->windowWidth, static_cast<long>(this->currentLocationX), static_cast<long>(this->currentLocationY), 0);

		convertNoiseToUchar3(this->canvas, field, this->windowWidth);

		glBindTexture(GL_TEXTURE_2D, this->textureId);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->windowWidth, this->windowHeight,
		                GL_RGB, GL_UNSIGNED_BYTE, reinterpret_cast<unsigned char *>(this->canvas));
	}

	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, this->textureId);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);  // Bottom-left
	glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);  // Bottom-right
	glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);  // Top-right
	glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);  // Top-left
	glEnd();

	// Swap front/back buffers (double buffering)
	glfwSwapBuffers(this->window);
}

void ProceduralPerlin::run()
{
	glfwSetWindowTitle(this->window, "Minecruft");

	while (!glfwWindowShouldClose(this->window)) {
		glfwPollEvents();
		renderField();
	}
}

static const unsigned WINDOW_WIDTH_MAX = 1024;
static const unsigned WINDOW_HEIGHT_MAX = 1024;

int main(int argc, char** argv)
{
	long seed = 0;

	if (argc > 1) {
		seed = atoi(argv[1]);
	}

	cudaSetDevice(0);

	try {
		ProceduralPerlin viewer(WINDOW_WIDTH_MAX, WINDOW_HEIGHT_MAX, seed);
		viewer.run();
	} catch (const std::exception& e) {
		fprintf(stderr, "Error: %s\n", e.what());
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
