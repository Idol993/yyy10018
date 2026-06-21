# TetGen WebAssembly

This directory contains the loader for TetGen WASM.

## Getting the Actual TetGen WASM Binary

The `tetgen.wasm` file is not included due to licensing considerations. To obtain a working TetGen WASM build:

### Option 1: Use an existing build
1. Check https://github.com/tomvanmele/tetgen.wasm for pre-built TetGen WASM
2. Download their `tetgen.wasm` file
3. Place it in this directory

### Option 2: Build from source
1. Get TetGen source from http://www.tetgen.org/
2. Compile to WASM using Emscripten:
```bash
emcc tetgen.c predicates.c -O3 -s EXPORTED_FUNCTIONS='["_tetrahedralize","_malloc","_free"]' \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s MODULARIZE=1 \
  -o tetgen.js
```
3. Place the resulting `tetgen.wasm` in this directory

### Expected WASM Exports

The WASM module should export:
- `tetrahedralize`: Main tetrahedralization function
- `malloc`: Memory allocation
- `free`: Memory deallocation
- `memory`: The WebAssembly Memory object

## Fallback Behavior

If `tetgen.wasm` is not present or cannot be loaded, the application will automatically
fall back to the built-in voxel-based mesher, with a clear visual indicator in the UI
that a lower-quality approximate mesh is being used.
