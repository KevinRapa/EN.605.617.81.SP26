
# Procedural Perlin Noise

## Description

This project implements a procedural perlin noise generator. Currently, the viewer displays
the perlin noise in grayscale. Supports mouse dragging to demonstrate procedural nature.

## Building and running

Built on Ubuntu 24

To build:

Requires CUDA runtime installed
Install GLFW3 and GLEW: `sudo apt-get install libglew-dev libglfw3 libglfw3-dev`

Run `make viewer` and drag the screen! Hope it works...

## Description of files

The code for the view is in src/main/frontend, and the perlin noise stuff is in src/main/backend

## TODOS

1. Implement multiple octaves
2. Create a 3D viewer in Python or something
3. Do something interesting like layering multiple perlin noise maps to create biomes
4. Make mouse dragging smoother (unlikely to happen)
