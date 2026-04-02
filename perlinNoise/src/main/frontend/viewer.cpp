#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <transformers.h>
#include <field.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

class PerlinGenerator {
private:
	GLFWwindow* window;                             // GLFW window handle
	GLuint texture_id;                              // OpenGL texture for display
	GLuint pbo_id;                                  // Pixel Buffer Object (unused but initialized)
	struct cudaGraphicsResource* cuda_pbo_resource; // CUDA-OpenGL interop handle

	int window_width;
	int window_height;
	double x_center;
	double y_center;
	bool mouse_dragging;
	double last_mouse_x;
	double last_mouse_y;
	bool needs_update;
	long seed;

public:
	PerlinGenerator(int width, int height, long seed);
	~PerlinGenerator();

	void initialize_opengl();
	void initialize_cuda_gl_interop();
	void setup_callbacks();

	static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
	static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos);
	static void framebuffer_size_callback(GLFWwindow* window, int width, int height);

	void handle_mouse_button(int button, int action, int mods);
	void handle_mouse_motion(double xpos, double ypos);
	void render();
	void run();
};

PerlinGenerator::PerlinGenerator(int width, int height, long seed)
{
	this->window_width = width;
	this->window_height = height;
	this->x_center = 0.0;
	this->y_center = 0.0;
	this->mouse_dragging = false;
	this->needs_update = true;
	this->seed = seed;

	initialize_opengl();
	initialize_cuda_gl_interop();
	setup_callbacks();
}

PerlinGenerator::~PerlinGenerator()
{
	if (cuda_pbo_resource) {
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
	}
	if (pbo_id) {
		glDeleteBuffers(1, &pbo_id);
	}
	if (texture_id) {
		glDeleteTextures(1, &texture_id);
	}
	if (window) {
		glfwDestroyWindow(window);
	}
	glfwTerminate();
}

void PerlinGenerator::initialize_opengl()
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
	window = glfwCreateWindow(window_width, window_height, "CUDA Fractal Explorer", NULL, NULL);
	if (!window) {
		fprintf(stderr, "Failed to create GLFW window!\n");
		glfwTerminate();
		exit(1);
	}

	// Make OpenGL context current for this thread
	glfwMakeContextCurrent(window);

	// DISABLE V-SYNC to show true GPU performance (not limited by monitor refresh)
	glfwSwapInterval(0);  // 0 = V-Sync OFF, 1 = V-Sync ON

	glfwSetWindowUserPointer(window, this);

	// Initialize GLEW to load OpenGL extensions
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW!\n");
		exit(1);
	}

	// Configure OpenGL state
	glDisable(GL_DEPTH_TEST);  // 2D rendering
	glViewport(0, 0, window_width, window_height);

	// Create texture to display fractal image
	glGenTextures(1, &texture_id);
	glBindTexture(GL_TEXTURE_2D, texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, window_width, window_height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

	glGenBuffers(1, &pbo_id);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, window_width * window_height * 3 * sizeof(unsigned char), nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void PerlinGenerator::initialize_cuda_gl_interop()
{
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo_id, cudaGraphicsMapFlagsWriteDiscard);
}

void PerlinGenerator::setup_callbacks()
{
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetCursorPosCallback(window, cursor_pos_callback);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
}

void PerlinGenerator::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	PerlinGenerator* viewer = static_cast<PerlinGenerator*>(glfwGetWindowUserPointer(window));
	viewer->handle_mouse_button(button, action, mods);
}

void PerlinGenerator::cursor_pos_callback(GLFWwindow* window, double xpos, double ypos)
{
	PerlinGenerator* viewer = static_cast<PerlinGenerator*>(glfwGetWindowUserPointer(window));
	viewer->handle_mouse_motion(xpos, ypos);
}

void PerlinGenerator::framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void PerlinGenerator::handle_mouse_button(int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_PRESS) {
			mouse_dragging = true;
			glfwGetCursorPos(window, &this->last_mouse_x, &this->last_mouse_y);
		} else if (action == GLFW_RELEASE) {
			mouse_dragging = false;
		}
	}
}

void PerlinGenerator::handle_mouse_motion(double xpos, double ypos)
{
	if (!mouse_dragging) {
		return;
	}

	// Calculate mouse movement delta in pixels
	double dx = xpos - this->last_mouse_x;
	double dy = ypos - this->last_mouse_y;

	this->x_center -= dx;
	this->y_center += dy;

	this->last_mouse_x = xpos;
	this->last_mouse_y = ypos;
	this->needs_update = true;
}

void PerlinGenerator::render()
{
	if (this->needs_update) {
		static uchar3* h_image = nullptr;
		if (!h_image) {
			h_image = (uchar3*)malloc(window_width * window_height * 3);
		}

		static float *field = nullptr;
		if (!field) {
			field = fieldAlloc(window_width);
		}

		createField(field, this->seed, window_width, this->x_center, this->y_center, 0);

		convertNoiseToUchar3(h_image, field, window_width);

		glBindTexture(GL_TEXTURE_2D, texture_id);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, window_width, window_height,
		                GL_RGB, GL_UNSIGNED_BYTE, reinterpret_cast<unsigned char *>(h_image));
	}

	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texture_id);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);  // Bottom-left
	glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);  // Bottom-right
	glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);  // Top-right
	glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);  // Top-left
	glEnd();

	// Swap front/back buffers (double buffering)
	glfwSwapBuffers(window);
}

void PerlinGenerator::run()
{
	glfwSetWindowTitle(window, "Minecruft");

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		render();
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
		PerlinGenerator viewer(WINDOW_WIDTH_MAX, WINDOW_HEIGHT_MAX, seed);
		viewer.run();
	} catch (const std::exception& e) {
		fprintf(stderr, "Error: %s\n", e.what());
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
