#!/bin/bash

# vulkan command line utilities, e.g. vulkaninfo and vkcube
sudo apt install -y vulkan-tools

# other dependencies
sudo apt install -y libxxf86vm-dev

# vulkan loader
sudo apt install -y libvulkan-dev

# vulkan validation layers and spir-v tools (for debugging)
sudo apt install -y vulkan-validationlayers-dev spirv-tools

# GLFW - platform indepenedent library for managing windows
sudo apt install -y libglfw3-dev

# GLM - library for linear algebra operations
sudo apt install -y libglm-dev

# glslc - glsl compiler - see https://github.com/google/shaderc/blob/main/downloads.md
