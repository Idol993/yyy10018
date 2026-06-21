import { SurfaceMesh, TetMesh } from './types';
import { generateTetMesh as generateVoxelTetMesh } from './tet-mesh';

export type MeshGeneratorType = 'tetgen' | 'voxel';

export interface MeshGeneratorResult {
  mesh: TetMesh;
  method: MeshGeneratorType;
  quality: {
    minQuality: number;
    avgQuality: number;
    maxAspectRatio: number;
  };
}

let tetgenWasmInstance: WebAssembly.Instance | null = null;
let tetgenMemory: WebAssembly.Memory | null = null;
let tetgenLoadPromise: Promise<boolean> | null = null;
let tetgenAvailable = false;
let tetgenLoadError: string | null = null;

export function getTetGenLoadError(): string | null {
  return tetgenLoadError;
}

export async function isTetGenAvailable(): Promise<boolean> {
  if (tetgenLoadPromise !== null) return tetgenLoadPromise;
  tetgenLoadPromise = loadTetGenWasm();
  return tetgenLoadPromise;
}

async function loadTetGenWasm(): Promise<boolean> {
  try {
    const wasmUrl = '/tetgen/tetgen.wasm';
    const response = await fetch(wasmUrl);
    if (!response.ok) {
      tetgenLoadError = `tetgen.wasm not found (HTTP ${response.status}) - place it in public/tetgen/`;
      return false;
    }

    const wasmBuffer = await response.arrayBuffer();

    const memory = new WebAssembly.Memory({ initial: 256, maximum: 2048 });
    tetgenMemory = memory;

    const importObject: WebAssembly.Imports = {
      env: {
        memory: memory,
        __memory_base: 0,
        __table_base: 0,
        __indirect_function_table: new WebAssembly.Table({ initial: 16, element: 'anyfunc' }),
        abort: () => { throw new Error('TetGen abort'); },
        _abort: () => { throw new Error('TetGen abort'); },
        _exit: () => {},
        abortOnCannotGrowMemory: () => { throw new Error('OOM'); },
        enlargeMemory: () => { memory.grow(64); },
        getTotalMemory: () => memory.buffer.byteLength,
        _emscripten_memcpy_big: (dest: number, src: number, len: number) => {
          const buf = new Uint8Array(memory.buffer);
          buf.copyWithin(dest, src, src + len);
        },
        _emscripten_resize_heap: (size: number) => {
          const needed = Math.ceil(size / 65536);
          const current = memory.buffer.byteLength / 65536;
          if (needed > current) {
            memory.grow(needed - current);
          }
          return true;
        },
        __lock: () => {},
        __unlock: () => {},
        ___cxa_allocate_exception: (size: number) => {
          const ptr = malloc(size + 16);
          return ptr + 16;
        },
        ___cxa_throw: () => { throw new Error('TetGen C++ exception'); },
        ___cxa_begin_catch: () => 0,
        ___cxa_end_catch: () => {},
        ___cxa_is_exception_type: () => 0,
        __setThrew: () => {},
        __syscall6: () => 0,
        __syscall54: () => 0,
        __syscall140: () => 0,
        _pthread_mutex_lock: () => 0,
        _pthread_mutex_unlock: () => 0,
        _pthread_cond_wait: () => 0,
        _pthread_cond_signal: () => 0,
        setTempRet0: () => {},
        getTempRet0: () => 0,
        _llvm_stacksave: () => 0,
        _llvm_stackrestore: () => {},
        _emscripten_stack_alloc: (size: number) => {
          return malloc(size);
        },
        _emscripten_stack_restore: () => {},
        _emscripten_stack_get_current: () => {
          return 0;
        },
      },
      'global.Math': Math as any,
      global: {
        NaN: NaN,
        Infinity: Infinity,
      },
    };

    let wasmInstance: WebAssembly.Instance;
    try {
      const { instance } = await WebAssembly.instantiate(wasmBuffer, importObject);
      wasmInstance = instance;
    } catch (compileErr: any) {
      tetgenLoadError = `WASM compile failed: ${compileErr?.message || String(compileErr)}`;
      return false;
    }

    if (typeof (wasmInstance.exports as any).tetrahedralize !== 'function' &&
        typeof (wasmInstance.exports as any)._tetrahedralize !== 'function') {
      tetgenLoadError = 'WASM loaded but missing tetrahedralize export';
      return false;
    }

    tetgenWasmInstance = wasmInstance;

    if (typeof (wasmInstance.exports as any)._initialize === 'function') {
      (wasmInstance.exports as any)._initialize();
    }
    if (typeof (wasmInstance.exports as any).___wasm_call_ctors === 'function') {
      (wasmInstance.exports as any).___wasm_call_ctors();
    }

    tetgenAvailable = true;
    return true;
  } catch (e: any) {
    tetgenLoadError = e?.message || String(e);
    tetgenAvailable = false;
    return false;
  }
}

function tryGetExport(name: string): any {
  if (!tetgenWasmInstance) return undefined;
  const exports = tetgenWasmInstance.exports as any;
  if (typeof exports[name] === 'function' || typeof exports[name] === 'number') {
    return exports[name];
  }
  const underscoreName = '_' + name;
  if (typeof exports[underscoreName] === 'function' || typeof exports[underscoreName] === 'number') {
    return exports[underscoreName];
  }
  return undefined;
}

function getExport(name: string): any {
  const exp = tryGetExport(name);
  if (exp === undefined) {
    throw new Error(`TetGen WASM export "${name}" (or "_${name}") not found`);
  }
  return exp;
}

function malloc(size: number): number {
  const fn = tryGetExport('malloc');
  if (typeof fn !== 'function') throw new Error('malloc not found');
  return fn(size);
}

function free(ptr: number): void {
  const fn = tryGetExport('free');
  if (typeof fn === 'function') fn(ptr);
}

function tetrahedralize(
  numPoints: number,
  points: Float64Array,
  numFaces: number,
  faceIndices: Int32Array,
  qualityRatio: number,
  maxVolume: number
): { numOutPoints: number; outPoints: Float64Array; numOutTets: number; outTets: Int32Array } {
  const mem = tetgenMemory!;

  const pointsPtr = malloc(numPoints * 3 * 8);
  const facesPtr = malloc(numFaces * 3 * 4);

  const pointsView = new Float64Array(mem.buffer, pointsPtr, numPoints * 3);
  pointsView.set(points);

  const facesView = new Int32Array(mem.buffer, facesPtr, numFaces * 3);
  facesView.set(faceIndices);

  const outNumPointsPtr = malloc(4);
  const outPointsPtrPtr = malloc(4);
  const outNumTetsPtr = malloc(4);
  const outTetsPtrPtr = malloc(4);

  const tetFn = tryGetExport('tetrahedralize');
  if (typeof tetFn !== 'function') {
    free(pointsPtr);
    free(facesPtr);
    free(outNumPointsPtr);
    free(outPointsPtrPtr);
    free(outNumTetsPtr);
    free(outTetsPtrPtr);
    throw new Error('tetrahedralize function not found in WASM exports');
  }

  let returnCode: number;
  try {
    returnCode = tetFn(
      numPoints, pointsPtr,
      numFaces, facesPtr,
      qualityRatio, maxVolume,
      outNumPointsPtr, outPointsPtrPtr,
      outNumTetsPtr, outTetsPtrPtr
    );
  } catch (e) {
    free(pointsPtr);
    free(facesPtr);
    free(outNumPointsPtr);
    free(outPointsPtrPtr);
    free(outNumTetsPtr);
    free(outTetsPtrPtr);
    throw new Error('TetGen tetrahedralize() threw: ' + (e as any)?.message);
  }

  if (returnCode !== 0) {
    free(pointsPtr);
    free(facesPtr);
    free(outNumPointsPtr);
    free(outPointsPtrPtr);
    free(outNumTetsPtr);
    free(outTetsPtrPtr);
    throw new Error(`TetGen tetrahedralize() returned error code ${returnCode}`);
  }

  const outNumPoints = new Int32Array(mem.buffer, outNumPointsPtr, 1)[0];
  const outPointsPtr = new Int32Array(mem.buffer, outPointsPtrPtr, 1)[0];
  const outNumTets = new Int32Array(mem.buffer, outNumTetsPtr, 1)[0];
  const outTetsPtr = new Int32Array(mem.buffer, outTetsPtrPtr, 1)[0];

  const outPoints = new Float64Array(mem.buffer, outPointsPtr, outNumPoints * 3).slice();
  const outTets = new Int32Array(mem.buffer, outTetsPtr, outNumTets * 4).slice();

  if (outPointsPtr) free(outPointsPtr);
  if (outTetsPtr) free(outTetsPtr);
  free(pointsPtr);
  free(facesPtr);
  free(outNumPointsPtr);
  free(outPointsPtrPtr);
  free(outNumTetsPtr);
  free(outTetsPtrPtr);

  return { numOutPoints: outNumPoints, outPoints, numOutTets: outNumTets, outTets };
}

export async function generateHighQualityTetMesh(
  surface: SurfaceMesh,
  targetSize: number,
  qualityRatio: number = 2.0
): Promise<MeshGeneratorResult> {
  const hasTetGen = await isTetGenAvailable();

  if (hasTetGen && tetgenWasmInstance) {
    try {
      return generateWithTetGen(surface, targetSize, qualityRatio);
    } catch (e: any) {
      console.warn('TetGen tetrahedralization failed, falling back to voxel method:', e);
      tetgenLoadError = 'Tetrahedralization failed: ' + (e?.message || String(e));
    }
  }

  const mesh = generateVoxelTetMesh(surface, targetSize);
  smoothMesh(mesh, 5);
  const quality = computeMeshQuality(mesh);
  return { mesh, method: 'voxel', quality };
}

function generateWithTetGen(
  surface: SurfaceMesh,
  targetSize: number,
  qualityRatio: number
): MeshGeneratorResult {
  const numVertices = surface.vertices.length / 3;
  const numFaces = surface.indices.length / 3;

  const points = new Float64Array(numVertices * 3);
  for (let i = 0; i < numVertices * 3; i++) {
    points[i] = surface.vertices[i];
  }

  const faceIndices = new Int32Array(numFaces * 3);
  for (let i = 0; i < numFaces * 3; i++) {
    faceIndices[i] = surface.indices[i];
  }

  const maxVolume = targetSize * targetSize * targetSize * 0.5;

  const result = tetrahedralize(
    numVertices,
    points,
    numFaces,
    faceIndices,
    qualityRatio,
    maxVolume
  );

  const numNodes = result.numOutPoints;
  const numElements = result.numOutTets;
  const nodes = result.outPoints;
  const elements = new Uint32Array(result.outTets);

  const { surfaceFaces, surfaceNormals, numSurfaceFaces } = extractSurfaceFaces(nodes, elements, numNodes, numElements);

  const mesh: TetMesh = {
    nodes,
    elements,
    surfaceFaces,
    surfaceNormals,
    numNodes,
    numElements,
    numSurfaceFaces,
  };

  const quality = computeMeshQuality(mesh);
  return { mesh, method: 'tetgen', quality };
}

function extractSurfaceFaces(
  nodes: Float64Array,
  elements: Uint32Array,
  numNodes: number,
  numElements: number
): { surfaceFaces: Uint32Array; surfaceNormals: Float32Array; numSurfaceFaces: number } {
  const faceMap = new Map<string, number>();
  const faceList: number[][] = [];
  const allFaces: number[][] = [];

  for (let e = 0; e < numElements; e++) {
    const n0 = elements[e * 4];
    const n1 = elements[e * 4 + 1];
    const n2 = elements[e * 4 + 2];
    const n3 = elements[e * 4 + 3];

    const tetFaces = [
      [n0, n1, n2],
      [n0, n2, n3],
      [n0, n3, n1],
      [n1, n3, n2],
    ];

    for (const f of tetFaces) {
      const sorted = [...f].sort((a, b) => a - b);
      const key = sorted.join(',');
      if (faceMap.has(key)) {
        faceMap.set(key, faceMap.get(key)! + 1);
      } else {
        faceMap.set(key, 1);
        allFaces.push(f);
      }
    }
  }

  for (const f of allFaces) {
    const sorted = [...f].sort((a, b) => a - b);
    const key = sorted.join(',');
    if (faceMap.get(key) === 1) {
      faceList.push(f);
    }
  }

  const numSurfaceFaces = faceList.length;
  const surfaceFaces = new Uint32Array(numSurfaceFaces * 3);
  const surfaceNormals = new Float32Array(numSurfaceFaces * 3);

  for (let i = 0; i < numSurfaceFaces; i++) {
    const f = faceList[i];
    surfaceFaces[i * 3] = f[0];
    surfaceFaces[i * 3 + 1] = f[1];
    surfaceFaces[i * 3 + 2] = f[2];

    const i0 = f[0] * 3;
    const i1 = f[1] * 3;
    const i2 = f[2] * 3;
    const ax = nodes[i1] - nodes[i0];
    const ay = nodes[i1 + 1] - nodes[i0 + 1];
    const az = nodes[i1 + 2] - nodes[i0 + 2];
    const bx = nodes[i2] - nodes[i0];
    const by = nodes[i2 + 1] - nodes[i0 + 1];
    const bz = nodes[i2 + 2] - nodes[i0 + 2];
    const nx = ay * bz - az * by;
    const ny = az * bx - ax * bz;
    const nz = ax * by - ay * bx;
    const nl = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (nl > 0) {
      surfaceNormals[i * 3] = nx / nl;
      surfaceNormals[i * 3 + 1] = ny / nl;
      surfaceNormals[i * 3 + 2] = nz / nl;
    }
  }

  return { surfaceFaces, surfaceNormals, numSurfaceFaces };
}

export function smoothMesh(mesh: TetMesh, iterations: number): void {
  const { nodes, elements, numNodes, numElements } = mesh;
  const nodeCount = new Uint32Array(numNodes);
  const newPositions = new Float64Array(numNodes * 3);

  const isBoundary = new Uint8Array(numNodes);
  for (let i = 0; i < mesh.numSurfaceFaces; i++) {
    isBoundary[mesh.surfaceFaces[i * 3]] = 1;
    isBoundary[mesh.surfaceFaces[i * 3 + 1]] = 1;
    isBoundary[mesh.surfaceFaces[i * 3 + 2]] = 1;
  }

  for (let iter = 0; iter < iterations; iter++) {
    nodeCount.fill(0);
    newPositions.fill(0);

    for (let e = 0; e < numElements; e++) {
      const n0 = elements[e * 4];
      const n1 = elements[e * 4 + 1];
      const n2 = elements[e * 4 + 2];
      const n3 = elements[e * 4 + 3];
      const nodeIds = [n0, n1, n2, n3];

      for (let ni = 0; ni < 4; ni++) {
        const nid = nodeIds[ni];
        if (isBoundary[nid]) continue;
        for (let nj = 0; nj < 4; nj++) {
          if (ni === nj) continue;
          const nid2 = nodeIds[nj];
          newPositions[nid * 3] += nodes[nid2 * 3];
          newPositions[nid * 3 + 1] += nodes[nid2 * 3 + 1];
          newPositions[nid * 3 + 2] += nodes[nid2 * 3 + 2];
          nodeCount[nid]++;
        }
      }
    }

    for (let i = 0; i < numNodes; i++) {
      if (isBoundary[i] || nodeCount[i] === 0) continue;
      const count = nodeCount[i];
      nodes[i * 3] = newPositions[i * 3] / count;
      nodes[i * 3 + 1] = newPositions[i * 3 + 1] / count;
      nodes[i * 3 + 2] = newPositions[i * 3 + 2] / count;
    }
  }
}

export function computeMeshQuality(mesh: TetMesh): {
  minQuality: number;
  avgQuality: number;
  maxAspectRatio: number;
} {
  let minQ = Infinity;
  let sumQ = 0;
  let maxAR = 0;
  let count = 0;

  const { nodes, elements, numElements } = mesh;

  for (let e = 0; e < numElements; e++) {
    const n0 = elements[e * 4] * 3;
    const n1 = elements[e * 4 + 1] * 3;
    const n2 = elements[e * 4 + 2] * 3;
    const n3 = elements[e * 4 + 3] * 3;

    const e1 = [
      nodes[n1] - nodes[n0],
      nodes[n1 + 1] - nodes[n0 + 1],
      nodes[n1 + 2] - nodes[n0 + 2],
    ];
    const e2 = [
      nodes[n2] - nodes[n0],
      nodes[n2 + 1] - nodes[n0 + 1],
      nodes[n2 + 2] - nodes[n0 + 2],
    ];
    const e3 = [
      nodes[n3] - nodes[n0],
      nodes[n3 + 1] - nodes[n0 + 1],
      nodes[n3 + 2] - nodes[n0 + 2],
    ];

    const volume = Math.abs(
      e1[0] * (e2[1] * e3[2] - e2[2] * e3[1]) -
      e1[1] * (e2[0] * e3[2] - e2[2] * e3[0]) +
      e1[2] * (e2[0] * e3[1] - e2[1] * e3[0])
    ) / 6;

    const l1 = Math.sqrt(e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2);
    const l2 = Math.sqrt(e2[0] ** 2 + e2[1] ** 2 + e2[2] ** 2);
    const l3 = Math.sqrt(e3[0] ** 2 + e3[1] ** 2 + e3[2] ** 2);
    const l4 = Math.sqrt(
      (nodes[n2] - nodes[n1]) ** 2 +
      (nodes[n2 + 1] - nodes[n1 + 1]) ** 2 +
      (nodes[n2 + 2] - nodes[n1 + 2]) ** 2
    );
    const l5 = Math.sqrt(
      (nodes[n3] - nodes[n1]) ** 2 +
      (nodes[n3 + 1] - nodes[n1 + 1]) ** 2 +
      (nodes[n3 + 2] - nodes[n1 + 2]) ** 2
    );
    const l6 = Math.sqrt(
      (nodes[n3] - nodes[n2]) ** 2 +
      (nodes[n3 + 1] - nodes[n2 + 1]) ** 2 +
      (nodes[n3 + 2] - nodes[n2 + 2]) ** 2
    );

    const maxEdge = Math.max(l1, l2, l3, l4, l5, l6);
    const minEdge = Math.min(l1, l2, l3, l4, l5, l6);
    const aspectRatio = maxEdge / Math.max(minEdge, 1e-12);

    const edgeLength = (l1 + l2 + l3 + l4 + l5 + l6) / 6;
    const idealVolume = (edgeLength ** 3) / (6 * Math.SQRT2);
    const quality = volume / Math.max(idealVolume, 1e-12);

    if (quality < minQ) minQ = quality;
    sumQ += quality;
    if (aspectRatio > maxAR) maxAR = aspectRatio;
    count++;
  }

  return {
    minQuality: minQ === Infinity ? 0 : minQ,
    avgQuality: count > 0 ? sumQ / count : 0,
    maxAspectRatio: maxAR,
  };
}
