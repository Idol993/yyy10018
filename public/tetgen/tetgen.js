/**
 * TetGen WebAssembly Loader
 * 
 * This is a standard Emscripten-compatible module loader for TetGen.
 * To use with real TetGen WASM:
 * 1. Obtain a TetGen WASM build (e.g., from https://github.com/tomvanmele/tetgen.wasm)
 * 2. Place tetgen.wasm in this same directory
 * 3. Ensure the WASM exports: tetrahedralize, malloc, free, etc.
 * 
 * The tetrahedralize function signature should be:
 *   tetrahedralize(
 *     numPoints: number, pointsPtr: number,
 *     numFaces: number, facesPtr: number,
 *     qualityRatio: number, maxVolume: number,
 *     outNumPointsPtr: number, outPointsPtrPtr: number,
 *     outNumTetsPtr: number, outTetsPtrPtr: number
 *   ): number
 */

var Module = (function() {
  var moduleOverrides = {};
  
  // Standard Emscripten module setup
  var Module = typeof Module !== 'undefined' ? Module : {};
  
  // Allow overrides from outside
  for (var key in moduleOverrides) {
    if (moduleOverrides.hasOwnProperty(key)) {
      Module[key] = moduleOverrides[key];
    }
  }
  
  var readyPromiseResolve, readyPromiseReject;
  Module.ready = new Promise(function(resolve, reject) {
    readyPromiseResolve = resolve;
    readyPromiseReject = reject;
  });
  
  var wasmBinaryFile = 'tetgen.wasm';
  var wasmBinary = null;
  
  // Function to load the WASM binary
  function getBinaryPromise() {
    if (wasmBinary) {
      return Promise.resolve(wasmBinary);
    }
    
    // Try to fetch the WASM file
    return fetch(wasmBinaryFile)
      .then(function(response) {
        if (!response.ok) {
          throw new Error('Failed to load tetgen.wasm: ' + response.status);
        }
        return response.arrayBuffer();
      })
      .then(function(buffer) {
        wasmBinary = new Uint8Array(buffer);
        return wasmBinary;
      });
  }
  
  // Memory management
  var HEAP8, HEAPU8, HEAP16, HEAPU16, HEAP32, HEAPU32, HEAPF32, HEAPF64;
  
  function updateMemoryViews() {
    var b = Module.HEAP8.buffer;
    HEAP8 = new Int8Array(b);
    HEAPU8 = new Uint8Array(b);
    HEAP16 = new Int16Array(b);
    HEAPU16 = new Uint16Array(b);
    HEAP32 = new Int32Array(b);
    HEAPU32 = new Uint32Array(b);
    HEAPF32 = new Float32Array(b);
    HEAPF64 = new Float64Array(b);
  }
  
  // Allocate memory
  Module._malloc = function(size) {
    if (!Module.wasmInstance) return 0;
    return Module.wasmInstance.exports.malloc(size);
  };
  
  Module._free = function(ptr) {
    if (!Module.wasmInstance) return;
    Module.wasmInstance.exports.free(ptr);
  };
  
  // Tetrahedralize wrapper
  Module.tetrahedralize = function(
    numPoints, pointsArray,
    numFaces, facesArray,
    qualityRatio, maxVolume
  ) {
    if (!Module.wasmInstance) {
      throw new Error('TetGen WASM not loaded');
    }
    
    // Allocate memory for input
    var pointsPtr = Module._malloc(numPoints * 3 * 8); // double
    var facesPtr = Module._malloc(numFaces * 4 * 4); // int32
    
    // Copy points (Float64)
    var pointsView = new Float64Array(Module.HEAPF64.buffer, pointsPtr, numPoints * 3);
    pointsView.set(pointsArray);
    
    // Copy faces (Int32)
    var facesView = new Int32Array(Module.HEAP32.buffer, facesPtr, numFaces * 4);
    facesView.set(facesArray);
    
    // Allocate memory for output pointers
    var outNumPointsPtr = Module._malloc(4);
    var outPointsPtrPtr = Module._malloc(4);
    var outNumTetsPtr = Module._malloc(4);
    var outTetsPtrPtr = Module._malloc(4);
    
    // Call WASM tetrahedralize
    var result = Module.wasmInstance.exports.tetrahedralize(
      numPoints, pointsPtr,
      numFaces, facesPtr,
      qualityRatio, maxVolume,
      outNumPointsPtr, outPointsPtrPtr,
      outNumTetsPtr, outTetsPtrPtr
    );
    
    if (result !== 0) {
      // Clean up on failure
      Module._free(pointsPtr);
      Module._free(facesPtr);
      Module._free(outNumPointsPtr);
      Module._free(outPointsPtrPtr);
      Module._free(outNumTetsPtr);
      Module._free(outTetsPtrPtr);
      throw new Error('Tetrahedralization failed with code: ' + result);
    }
    
    // Read output counts
    var outNumPoints = Module.HEAP32[outNumPointsPtr >> 2];
    var outPointsPtr = Module.HEAP32[outPointsPtrPtr >> 2];
    var outNumTets = Module.HEAP32[outNumTetsPtr >> 2];
    var outTetsPtr = Module.HEAP32[outTetsPtrPtr >> 2];
    
    // Copy output data
    var outPoints = new Float64Array(Module.HEAPF64.buffer, outPointsPtr, outNumPoints * 3).slice();
    var outTets = new Int32Array(Module.HEAP32.buffer, outTetsPtr, outNumTets * 4).slice();
    
    // Free WASM-allocated memory
    Module.wasmInstance.exports.free(outPointsPtr);
    Module.wasmInstance.exports.free(outTetsPtr);
    
    // Free our allocations
    Module._free(pointsPtr);
    Module._free(facesPtr);
    Module._free(outNumPointsPtr);
    Module._free(outPointsPtrPtr);
    Module._free(outNumTetsPtr);
    Module._free(outTetsPtrPtr);
    
    return {
      numPoints: outNumPoints,
      points: outPoints,
      numTets: outNumTets,
      tets: outTets
    };
  };
  
  // Initialize WASM
  function instantiateWasm(imports) {
    return getBinaryPromise().then(function(binary) {
      return WebAssembly.instantiate(binary, imports);
    }).then(function(instance) {
      Module.wasmInstance = instance.instance;
      Module.HEAP8 = new Int8Array(instance.instance.exports.memory.buffer);
      Module.HEAPU8 = new Uint8Array(instance.instance.exports.memory.buffer);
      Module.HEAP16 = new Int16Array(instance.instance.exports.memory.buffer);
      Module.HEAPU16 = new Uint16Array(instance.instance.exports.memory.buffer);
      Module.HEAP32 = new Int32Array(instance.instance.exports.memory.buffer);
      Module.HEAPU32 = new Uint32Array(instance.instance.exports.memory.buffer);
      Module.HEAPF32 = new Float32Array(instance.instance.exports.memory.buffer);
      Module.HEAPF64 = new Float64Array(instance.instance.exports.memory.buffer);
      updateMemoryViews();
      
      // Call malloc/free wrapper setup
      Module._malloc = function(size) {
        return instance.instance.exports.malloc(size);
      };
      Module._free = function(ptr) {
        instance.instance.exports.free(ptr);
      };
      
      readyPromiseResolve(Module);
      return instance.instance;
    }).catch(function(err) {
      console.error('Failed to load TetGen WASM:', err);
      readyPromiseReject(err);
    });
  }
  
  // Try to initialize
  var info = {
    'env': {
      '__memory_base': 0,
      '__table_base': 0,
      'memory': new WebAssembly.Memory({ initial: 256, maximum: 1024 }),
      'table': new WebAssembly.Table({ initial: 14, element: 'anyfunc' }),
      '__indirect_function_table': new WebAssembly.Table({ initial: 14, element: 'anyfunc' }),
      'abort': function() { throw new Error('abort'); },
      'enlargeMemory': function() { return 0; },
      'getTotalMemory': function() { return 16777216; },
      'abortOnCannotGrowMemory': function() { throw new Error('OOM'); },
    },
    'global': {
      'NaN': NaN,
      'Infinity': Infinity
    },
    'global.Math': Math,
  };
  
  // Attempt WASM loading
  instantiateWasm(info).catch(function(err) {
    // Silent catch - availability is checked via isTetGenAvailable
  });
  
  return Module;
})();

// Export for ES module usage
if (typeof exports === 'object' && typeof module === 'object') {
  module.exports = Module;
} else if (typeof define === 'function' && define['amd']) {
  define([], function() { return Module; });
}

// Make available globally
if (typeof window !== 'undefined') {
  window.TetGen = Module;
}
