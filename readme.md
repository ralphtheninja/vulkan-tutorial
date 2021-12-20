# vulkan-tutorial

My code following along https://vulkan-tutorial.com/

## usage

Build the code: `make`

Run the executable: `make test` or `./vulkantest`

Develop: `make dev` (watch files and recompile automatically)

## vulkan loader

Debug the vulkan loader: `VK_LOADER_DEBUG=all ./vulkantest` (can replace `all` with `warn`, `error`, `info`, `perf` or `debug`)

Force a particular driver path:

`VK_DRIVERS_PATH` -> delimited list of paths to location of driver JSON files

Force a particular ICD:

`VK_ICD_FILENAMES` -> delimited list of specific driver JSON files

Force on a Layer from outside the application:

`VK_INSTANCE_LAYERS` -> delimited list of layer names to enable

Force/override the layer path:

`VK_LAYER_PATH` -> delimited list of paths to search for layer JSON files
