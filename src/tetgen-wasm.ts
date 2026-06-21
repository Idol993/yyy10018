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

export interface TetGenModule {
  ready: Promise<any>;
  wasmInstance: any;
  tetrahedralize: (
    numPoints: number, points: Float64Array,
    numFaces: number, faces: Int32Array,
    qualityRatio: number, maxVolume: number
  ) => {
    numPoints: number;
    points: Float64Array;
    numTets: number;
    tets: Int32Array;
  };
  _malloc: (size: number) => number;
  _free: (ptr: number) => void;
}

let tetgenModule: TetGenModule | null = null;
let tetgenLoadPromise: Promise<boolean> | null = null;
let tetgenAvailable = false;
let tetgenLoadError: string | null = null;

export function getTetGenLoadError(): string | null {
  return tetgenLoadError;
}

export async function isTetGenAvailable(): Promise<boolean> {
  if (tetgenLoadPromise !== null) return tetgenLoadPromise;
  tetgenLoadPromise = loadTetGen();
  return tetgenLoadPromise;
}

async function loadTetGen(): Promise<boolean> {
  try {
    const existingModule = (window as any).TetGen as TetGenModule;
    if (existingModule && typeof existingModule.tetrahedralize === 'function') {
      if (existingModule.ready) {
        try {
          await existingModule.ready;
          if (existingModule.wasmInstance) {
            tetgenModule = existingModule;
            tetgenAvailable = true;
            return true;
          }
        } catch (e: any) {
          tetgenLoadError = 'WASM init failed: ' + (e?.message || String(e));
        }
      } else {
        tetgenModule = existingModule;
        tetgenAvailable = true;
        return true;
      }
    }

    try {
      const response = await fetch('/tetgen/tetgen.js', { cache: 'no-store' });
      if (!response.ok) {
        tetgenLoadError = `Cannot load tetgen.js (HTTP ${response.status})`;
        throw new Error(tetgenLoadError);
      }

      const scriptText = await response.text();
      const script = document.createElement('script');
      script.textContent = scriptText;
      document.head.appendChild(script);

      await new Promise<void>((resolve) => setTimeout(resolve, 50));

      const module = (window as any).TetGen as TetGenModule;
      if (!module) {
        tetgenLoadError = 'tetgen.js did not expose TetGen module';
        throw new Error(tetgenLoadError);
      }

      if (module.ready) {
        try {
          await module.ready;
          if (module.wasmInstance) {
            tetgenModule = module;
            tetgenAvailable = true;
            return true;
          } else {
            tetgenLoadError = 'tetgen.wasm not found or failed to compile';
            return false;
          }
        } catch (e: any) {
          tetgenLoadError = 'WASM load failed: ' + (e?.message || String(e));
          return false;
        }
      } else {
        if (typeof module.tetrahedralize === 'function') {
          tetgenModule = module;
          tetgenAvailable = true;
          return true;
        }
        tetgenLoadError = 'TetGen module missing tetrahedralize function';
        return false;
      }
    } catch (e: any) {
      if (!tetgenLoadError) {
        tetgenLoadError = e?.message || String(e);
      }
      return false;
    }
  } catch (e: any) {
    tetgenLoadError = e?.message || String(e);
    tetgenAvailable = false;
    return false;
  }
}

export async function generateHighQualityTetMesh(
  surface: SurfaceMesh,
  targetSize: number,
  qualityRatio: number = 2.0
): Promise<MeshGeneratorResult> {
  const hasTetGen = await isTetGenAvailable();

  if (hasTetGen && tetgenModule) {
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
  if (!tetgenModule) {
    throw new Error('TetGen module not loaded');
  }

  const numVertices = surface.vertices.length / 3;
  const numFaces = surface.indices.length / 3;

  const facesWithOrientation = new Int32Array(numFaces * 4);
  for (let i = 0; i < numFaces; i++) {
    facesWithOrientation[i * 4] = surface.indices[i * 3];
    facesWithOrientation[i * 4 + 1] = surface.indices[i * 3 + 1];
    facesWithOrientation[i * 4 + 2] = surface.indices[i * 3 + 2];
    facesWithOrientation[i * 4 + 3] = 0;
  }

  const maxVolume = targetSize * targetSize * targetSize * 0.5;

  const result = tetgenModule.tetrahedralize(
    numVertices,
    surface.vertices as unknown as Float64Array,
    numFaces,
    facesWithOrientation,
    qualityRatio,
    maxVolume
  );

  const numNodes = result.numPoints;
  const numElements = result.numTets;
  const nodes = new Float64Array(result.points);
  const elements = new Uint32Array(result.tets);

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
