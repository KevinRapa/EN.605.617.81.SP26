// interactive_viewer.cpp - Real-time Interactive Fractal Viewer
// 
// High-performance fractal explorer using CUDA for GPU-accelerated computation
// and OpenGL for real-time rendering. Supports multiple fractal types, color
// schemes, and interactive navigation with mouse/keyboard controls.
//
// Performance: ~900+ FPS @ 1920x1080 on RTX Pro 6000 (Blackwell) (iteration 64)

// OpenGL/GLFW libraries for windowing and rendering
#include <GL/glew.h>             // OpenGL Extension Wrangler - modern OpenGL features
#include <GLFW/glfw3.h>          // Cross-platform windowing and input handling

// CUDA libraries for GPU computation
#include <cuda_runtime.h>       // CUDA runtime API
#include <cuda_gl_interop.h>    // CUDA-OpenGL interoperability (PBO support)

// Standard C/C++ libraries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <chrono>              // High-resolution timing for FPS counter
#include <cstring>

// ============================================================================
// Data Structures
// ============================================================================

// Supported fractal types - each has unique iteration formula
typedef enum {
    MANDELBROT = 0,    // z = z² + c, starting at z=0
    JULIA = 1,         // z = z² + c, starting at z=pixel, constant c
    BURNING_SHIP = 2,  // Uses absolute values: z = (|Re(z)|+i|Im(z)|)² + c
    TRICORN = 3        // Uses complex conjugate: z = z̄² + c
} FractalType;

// Color schemes for visualizing iteration counts
typedef enum {
    GRAYSCALE = 0,     // Simple grayscale gradient
    HSV_RAINBOW = 1,   // Full HSV color wheel (animated)
    FIRE = 2,          // Red → yellow → white progression
    OCEAN = 3,         // Blue gradient (dark → light)
    PSYCHEDELIC = 4,   // Animated sine wave colors
    ELECTRIC = 5       // Blue → white intensity ramp
} ColorScheme;

// Parameters passed to CUDA kernel for fractal computation
struct FractalParams {
    double x_center, y_center;  // View center in complex plane
    double zoom;                // Zoom level (1.0 = default view)
    double julia_cx, julia_cy;  // Julia set constant (only for JULIA type)
    int max_iterations;         // Escape iteration limit (64-2048)
    FractalType type;           // Which fractal algorithm to use
    ColorScheme color_scheme;   // How to color the output
    double animation_time;      // Time parameter for animated effects
};

// External CUDA kernel launcher (implemented in fractal_engine.cu)
extern "C" void launch_fractal_kernel(unsigned char* d_image, int width, int height, FractalParams params);

// ============================================================================
// Interactive Fractal Viewer Class
// ============================================================================
// Manages OpenGL window, CUDA computation, and user interaction for real-time
// fractal exploration. Uses GLFW for windowing and OpenGL for rendering.
//
class InteractiveFractalViewer {
private:
    // OpenGL/GLFW resources
    GLFWwindow* window;                             // GLFW window handle
    GLuint texture_id;                              // OpenGL texture for display
    GLuint pbo_id;                                  // Pixel Buffer Object (unused but initialized)
    struct cudaGraphicsResource* cuda_pbo_resource; // CUDA-OpenGL interop handle
    
    // Window dimensions
    int window_width, window_height;
    
    // Fractal view state - defines what region of complex plane to render
    double x_center, y_center;  // Center point in complex plane
    double zoom;                // Magnification factor (higher = more zoomed in)
    double julia_cx, julia_cy;  // Julia set parameters (only used for JULIA type)
    int max_iterations;         // Maximum escape iterations (affects detail)
    int fractal_type;           // Current fractal (0-3, see FractalType enum)
    int color_scheme;           // Current color mapping (0-5, see ColorScheme enum)
    double animation_time;      // Accumulator for animation effects
    
    // Mouse interaction state
    bool mouse_dragging;                // True when left button is pressed
    double last_mouse_x, last_mouse_y;  // Previous mouse position for delta calculation
    
    // Rendering state
    bool auto_animate;          // Continuous animation mode (toggled by Space)
    bool needs_update;          // Flag to trigger frame re-render
    bool benchmark_mode;        // Force continuous recomputation for true FPS measurement
    
    // Performance monitoring
    std::chrono::high_resolution_clock::time_point last_frame_time;
    float fps;                  // Frames per second (updated once per second)
    int frame_count;            // Frame counter for FPS calculation
    
    // Print control menu to console
    void print_menu() {
        printf("\n=== CUDA Fractal Explorer ===\n");
        printf("Controls: Mouse drag=pan, wheel=zoom | 1-4=fractals | Q/W/E/R/T/Y=colors\n");
        printf("          +/- =iterations | 0=min iter (64) | 9=max iter (2048) | Space=animate\n");
        printf("          Arrows=pan | C=center | V=reset zoom | X=reset view/colors/animation\n");
        printf("          B=benchmark mode (shows TRUE computational FPS)\n");
        printf("          M=show menu | J/K/I/L=Julia params | ESC=exit\n\n");
    }
    
public:
    // Constructor - initializes viewer with specified window dimensions
    // Default view: Mandelbrot set centered at (-0.5, 0), zoom 1.0, 256 iterations
    InteractiveFractalViewer(int width, int height, bool start_benchmark = false, bool start_animate = false) 
        : window_width(width), window_height(height),
          x_center(-0.5), y_center(0.0), zoom(1.0),
          julia_cx(-0.7), julia_cy(0.27015),
          max_iterations(256), fractal_type(0), color_scheme(1),
          animation_time(0.0), mouse_dragging(false),
          auto_animate(start_animate), needs_update(true), benchmark_mode(start_benchmark), fps(0.0f), frame_count(0) {
        
        initialize_opengl();
        initialize_cuda_gl_interop();
        setup_callbacks();
        
        last_frame_time = std::chrono::high_resolution_clock::now();
        
        print_menu();
        
        if (start_benchmark) {
            printf("\n*** BENCHMARK MODE ENABLED from command line ***\n");
            printf("Showing COMPUTATIONAL FPS (forces recomputation every frame)\n\n");
        }
        if (start_animate) {
            printf("\n*** AUTO-ANIMATION ENABLED from command line ***\n");
            printf("Color animation active (press Space to toggle)\n\n");
        }
    }
    
    ~InteractiveFractalViewer() {
        cleanup();
    }
    
    // ========================================================================
    // Initialization Methods
    // ========================================================================
    
    // Initialize OpenGL context, window, and rendering resources
    void initialize_opengl() {
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
        window = glfwCreateWindow(window_width, window_height, 
                                 "CUDA Fractal Explorer", NULL, NULL);
        if (!window) {
            fprintf(stderr, "Failed to create GLFW window!\n");
            glfwTerminate();
            exit(1);
        }
        
        // Make OpenGL context current for this thread
        glfwMakeContextCurrent(window);
        
        // DISABLE V-SYNC to show true GPU performance (not limited by monitor refresh)
        glfwSwapInterval(0);  // 0 = V-Sync OFF, 1 = V-Sync ON
        
        glfwSetWindowUserPointer(window, this);  // Store 'this' for callbacks
        
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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, window_width, window_height, 
                     0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        
        glGenBuffers(1, &pbo_id);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, 
                     window_width * window_height * 3 * sizeof(unsigned char), 
                     nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
    
    // Register PBO with CUDA for potential zero-copy rendering
    void initialize_cuda_gl_interop() {
        cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo_id, 
                                   cudaGraphicsMapFlagsWriteDiscard);
    }
    
    // Register GLFW input callbacks
    void setup_callbacks() {
        glfwSetKeyCallback(window, key_callback);
        glfwSetMouseButtonCallback(window, mouse_button_callback);
        glfwSetCursorPosCallback(window, cursor_pos_callback);
        glfwSetScrollCallback(window, scroll_callback);
        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    }
    
    // ========================================================================
    // GLFW Callback Functions (static wrappers)
    // ========================================================================
    // GLFW callbacks must be static functions, so we retrieve the instance
    // pointer and forward to member functions
    
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        InteractiveFractalViewer* viewer = static_cast<InteractiveFractalViewer*>(
            glfwGetWindowUserPointer(window));
        viewer->handle_keyboard(key, scancode, action, mods);
    }
    
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        InteractiveFractalViewer* viewer = static_cast<InteractiveFractalViewer*>(
            glfwGetWindowUserPointer(window));
        viewer->handle_mouse_button(button, action, mods);
    }
    
    static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
        InteractiveFractalViewer* viewer = static_cast<InteractiveFractalViewer*>(
            glfwGetWindowUserPointer(window));
        viewer->handle_mouse_motion(xpos, ypos);
    }
    
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        InteractiveFractalViewer* viewer = static_cast<InteractiveFractalViewer*>(
            glfwGetWindowUserPointer(window));
        viewer->handle_scroll(xoffset, yoffset);
    }
    
    static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
    }
    
    // ========================================================================
    // Input Handling
    // ========================================================================
    
    // Handle keyboard input - fractal navigation, settings, and controls
    void handle_keyboard(int key, int scancode, int action, int mods) {
        if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
        
        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
                
            case GLFW_KEY_1: fractal_type = 0; needs_update = true; printf("Mandelbrot\n"); break;
            case GLFW_KEY_2: fractal_type = 1; needs_update = true; printf("Julia\n"); break;
            case GLFW_KEY_3: fractal_type = 2; needs_update = true; printf("Burning Ship\n"); break;
            case GLFW_KEY_4: fractal_type = 3; needs_update = true; printf("Tricorn\n"); break;
            
            case GLFW_KEY_0: max_iterations = 64; needs_update = true; printf("Iterations: %d (min)\n", max_iterations); break;
            case GLFW_KEY_9: max_iterations = 2048; needs_update = true; printf("Iterations: %d (max)\n", max_iterations); break;
            
            case GLFW_KEY_Q: color_scheme = 0; needs_update = true; printf("Key: Q | Color: GRAYSCALE\n"); break;
            case GLFW_KEY_W: color_scheme = 1; needs_update = true; printf("Key: W | Color: HSV_RAINBOW\n"); break;
            case GLFW_KEY_E: color_scheme = 2; needs_update = true; printf("Key: E | Color: FIRE\n"); break;
            case GLFW_KEY_R: color_scheme = 3; needs_update = true; printf("Key: R | Color: OCEAN\n"); break;
            case GLFW_KEY_T: color_scheme = 4; needs_update = true; printf("Key: T | Color: PSYCHEDELIC\n"); break;
            case GLFW_KEY_Y: color_scheme = 5; needs_update = true; printf("Key: Y | Color: ELECTRIC\n"); break;
            
            case GLFW_KEY_EQUAL:
            case GLFW_KEY_KP_ADD:
                max_iterations = std::min(2048, max_iterations + 64);
                needs_update = true;
                printf("Iterations: %d\n", max_iterations);
                break;
                
            case GLFW_KEY_MINUS:
            case GLFW_KEY_KP_SUBTRACT:
                max_iterations = std::max(64, max_iterations - 64);
                needs_update = true;
                printf("Iterations: %d\n", max_iterations);
                break;
                
            case GLFW_KEY_SPACE:
                auto_animate = !auto_animate;
                printf("Animation: %s\n", auto_animate ? "ON" : "OFF");
                break;
                
            case GLFW_KEY_B:
                benchmark_mode = !benchmark_mode;
                printf("Benchmark Mode: %s - Showing %s FPS\n", 
                       benchmark_mode ? "ON" : "OFF",
                       benchmark_mode ? "COMPUTATIONAL" : "DISPLAY");
                if (benchmark_mode) {
                    printf("WARNING: This forces recomputation every frame (lower FPS but shows true GPU performance)\n");
                }
                break;
                
            case GLFW_KEY_M:
                print_menu();
                break;
                
            case GLFW_KEY_C:
                x_center = (fractal_type == 0 || fractal_type == 3) ? -0.5 : 0.0;
                y_center = 0.0;
                needs_update = true;
                printf("Centered\n");
                break;
                
            case GLFW_KEY_V:
                zoom = 1.0;
                needs_update = true;
                printf("Zoom reset\n");
                break;
                
            case GLFW_KEY_UP: y_center += 0.1 / zoom; needs_update = true; break;
            case GLFW_KEY_DOWN: y_center -= 0.1 / zoom; needs_update = true; break;
            case GLFW_KEY_LEFT: x_center -= 0.1 / zoom; needs_update = true; break;
            case GLFW_KEY_RIGHT: x_center += 0.1 / zoom; needs_update = true; break;
            
            case GLFW_KEY_X:
                // Full reset (keeps current fractal type and iterations)
                x_center = (fractal_type == 0 || fractal_type == 3) ? -0.5 : 0.0;  // Center based on fractal
                y_center = 0.0;
                zoom = 1.0;
                // fractal_type unchanged - stay on current fractal
                color_scheme = 1;       // Rainbow (HSV)
                // max_iterations unchanged - keep current detail level
                julia_cx = -0.7;        // Reset Julia parameters
                julia_cy = 0.27015;
                auto_animate = false;   // Turn off animation
                animation_time = 0.0;   // Reset animation time
                needs_update = true;
                printf("Reset view/colors/animation\n");
                break;
                
            case GLFW_KEY_J:
                if (fractal_type == 1) { julia_cx -= 0.01; needs_update = true; }
                break;
            case GLFW_KEY_L:
                if (fractal_type == 1) { julia_cx += 0.01; needs_update = true; }
                break;
            case GLFW_KEY_K:
                if (fractal_type == 1) { julia_cy -= 0.01; needs_update = true; }
                break;
            case GLFW_KEY_I:
                if (fractal_type == 1) { julia_cy += 0.01; needs_update = true; }
                break;
        }
    }
    
    // Handle mouse button events - start/stop dragging for panning
    void handle_mouse_button(int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            if (action == GLFW_PRESS) {
                mouse_dragging = true;
                glfwGetCursorPos(window, &last_mouse_x, &last_mouse_y);
            } else if (action == GLFW_RELEASE) {
                mouse_dragging = false;
            }
        }
    }
    
    // Handle mouse motion - pan view when dragging
    void handle_mouse_motion(double xpos, double ypos) {
        if (!mouse_dragging) return;
        
        // Calculate mouse movement delta in pixels
        double dx = xpos - last_mouse_x;
        double dy = ypos - last_mouse_y;
        
        // Convert pixel delta to complex plane coordinates
        // range = width of visible complex plane
        double range = 4.0 / zoom;
        x_center -= (dx / window_width) * range;  // Negative for natural drag direction
        y_center += (dy / window_height) * range * (double)window_height / window_width;  // Positive for inverted Y
        
        // Update last position for next delta calculation
        last_mouse_x = xpos;
        last_mouse_y = ypos;
        needs_update = true;
    }
    
    // Handle mouse scroll - zoom in/out while keeping point under cursor stationary
    void handle_scroll(double xoffset, double yoffset) {
        double mouse_x, mouse_y;
        glfwGetCursorPos(window, &mouse_x, &mouse_y);
        
        // Step 1: Calculate complex plane coordinate under mouse cursor BEFORE zoom
        double range = 4.0 / zoom;
        double aspect = (double)window_height / window_width;
        double fractal_x = x_center - range + (range * 2.0 * mouse_x) / window_width;
        double fractal_y = y_center + range * aspect - (range * aspect * 2.0 * mouse_y) / window_height;
        
        // Step 2: Apply zoom (1.2x in or out)
        double zoom_factor = (yoffset > 0) ? 1.2 : (1.0 / 1.2);
        zoom *= zoom_factor;
        
        // Step 3: Adjust center so that fractal_x,fractal_y stays under mouse cursor
        // This creates the "zoom to cursor" effect like in Google Maps
        double new_range = 4.0 / zoom;
        x_center = fractal_x + new_range - (new_range * 2.0 * mouse_x) / window_width;
        y_center = fractal_y - new_range * aspect + (new_range * aspect * 2.0 * mouse_y) / window_height;
        
        needs_update = true;
        printf("Zoom: %.2e\n", zoom);
    }
    
    // ========================================================================
    // Rendering
    // ========================================================================
    
    // Render one frame - compute fractal on GPU and display result
    void render_frame() {
        // Update animation time if auto-animate is enabled
        if (auto_animate) {
            animation_time += 0.016;  // ~60 FPS animation speed
            needs_update = true;
        }
        
        // Only recompute if something changed (zoom, pan, settings, etc.)
        if (needs_update) {
            // Allocate host memory for image transfer (static = allocated once)
            static unsigned char* h_image = nullptr;
            if (!h_image) {
                h_image = (unsigned char*)malloc(window_width * window_height * 3);
            }
            
            // Allocate device memory for fractal computation (static = allocated once)
            static unsigned char* d_image = nullptr;
            if (!d_image) {
                cudaMalloc(&d_image, window_width * window_height * 3);
            }
            
            // Package all fractal parameters
            FractalParams params;
            params.x_center = x_center;
            params.y_center = y_center;
            params.zoom = zoom;
            params.julia_cx = julia_cx;
            params.julia_cy = julia_cy;
            params.max_iterations = max_iterations;
            params.type = (FractalType)fractal_type;
            params.color_scheme = (ColorScheme)color_scheme;
            params.animation_time = animation_time;
            
            // Step 1: Launch CUDA kernel to compute fractal on GPU
            launch_fractal_kernel(d_image, window_width, window_height, params);
            
            // Step 2: Copy result from GPU to CPU (blocking transfer)
            cudaMemcpy(h_image, d_image, window_width * window_height * 3, cudaMemcpyDeviceToHost);
            
            // Step 3: Upload CPU buffer to OpenGL texture
            glBindTexture(GL_TEXTURE_2D, texture_id);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, window_width, window_height,
                            GL_RGB, GL_UNSIGNED_BYTE, h_image);
            
            // In benchmark mode, force recomputation every frame to measure true computational FPS
            needs_update = benchmark_mode;
        }
        
        // Render full-screen textured quad with fractal image
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        
        // Draw quad using legacy OpenGL (compatibility profile)
        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);  // Bottom-left
        glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);  // Bottom-right
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);  // Top-right
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);  // Top-left
        glEnd();
        
        // Swap front/back buffers (double buffering)
        glfwSwapBuffers(window);
    }
    
    // Update FPS counter and window title (once per second)
    void update_fps() {
        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - last_frame_time);
        
        // Update window title with FPS every 1000ms
        if (duration.count() >= 1000) {
            fps = frame_count / (duration.count() / 1000.0f);
            frame_count = 0;
            last_frame_time = current_time;
            
            // Display FPS, zoom level, and iteration count in title bar
            char title[256];
            snprintf(title, sizeof(title), 
                     "CUDA Fractal Explorer - %.0f %s FPS | Zoom: %.2e | Iter: %d", 
                     fps, benchmark_mode ? "COMPUTE" : "DISPLAY", zoom, max_iterations);
            glfwSetWindowTitle(window, title);
        }
    }
    
    // Main event loop - runs until window is closed
    void run() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();    // Process input events (keyboard, mouse, etc.)
            render_frame();      // Compute and display fractal
            update_fps();        // Update performance counter
        }
    }
    
    // Cleanup resources before exit
    void cleanup() {
        // Unregister CUDA-OpenGL interop
        if (cuda_pbo_resource) {
            cudaGraphicsUnregisterResource(cuda_pbo_resource);
        }
        // Delete OpenGL resources
        if (pbo_id) {
            glDeleteBuffers(1, &pbo_id);
        }
        if (texture_id) {
            glDeleteTextures(1, &texture_id);
        }
        // Destroy window and terminate GLFW
        if (window) {
            glfwDestroyWindow(window);
        }
        glfwTerminate();
    }
};

// ============================================================================
// Main Entry Point
// ============================================================================

// Print usage information
void print_usage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  --benchmark         Enable benchmark mode on startup (force recompute each frame)\n");
    printf("  --animate           Enable automatic zoom animation on startup\n");
    printf("  --resolution W H    Set window resolution (default: 1280x720)\n");
    printf("                      W and H must be positive integers\n");
    printf("  --help, -h          Show this help message\n");
    printf("\nExamples:\n");
    printf("  %s --resolution 1920 1080\n", program_name);
    printf("  %s --benchmark --resolution 1920 1080\n", program_name);
    printf("  %s --animate\n", program_name);
}

int main(int argc, char** argv) {
    // Default settings
    int width = 1280, height = 720;
    bool start_benchmark = false;
    bool start_animate = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--resolution") == 0) {
            // Check if we have two more arguments
            if (i + 2 >= argc) {
                fprintf(stderr, "Error: --resolution requires two arguments (width height)\n");
                print_usage(argv[0]);
                return 1;
            }
            
            // Parse width and height
            width = atoi(argv[i + 1]);
            height = atoi(argv[i + 2]);
            
            // Validate parsed values
            if (width <= 0 || height <= 0) {
                fprintf(stderr, "Error: Invalid resolution %dx%d. Width and height must be positive integers.\n", 
                        width, height);
                return 1;
            }
            
            // Check reasonable bounds (optional but recommended)
            if (width < 320 || height < 240) {
                fprintf(stderr, "Error: Resolution too small. Minimum is 320x240.\n");
                return 1;
            }
            if (width > 7680 || height > 4320) {
                fprintf(stderr, "Warning: Resolution %dx%d is very large and may cause performance issues.\n",
                        width, height);
            }
            
            i += 2;  // Skip the next two arguments
        }
        else if (strcmp(argv[i], "--benchmark") == 0) {
            start_benchmark = true;
        }
        else if (strcmp(argv[i], "--animate") == 0) {
            start_animate = true;
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        else {
            fprintf(stderr, "Error: Unknown argument '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Select GPU 0 (primary CUDA device)
    cudaSetDevice(0);
    
    try {
        // Create viewer and run main loop
        InteractiveFractalViewer viewer(width, height, start_benchmark, start_animate);
        viewer.run();
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    
    return 0;
}
