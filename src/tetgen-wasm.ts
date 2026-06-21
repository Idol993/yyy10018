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

let tetgenModule: any = null;
let tetgenLoadPromise: Promise<boolean> | null = null;
let tetgenAvailable = false;

export async function isTetGenAvailable(): Promise<boolean> {
  if (tetgenLoadPromise !== null) return tetgenLoadPromise;
  tetgenLoadPromise = loadTetGen();
  return tetgenLoadPromise;
}

async function loadTetGen(): Promise<boolean> {
  try {
    const Module = (window as any).TetGen;
    if (Module && typeof Module.tetrahedralize === 'function') {
      tetgenModule = Module;
      tetgenAvailable = true;
      return true;
    }

    try {
      const response = await fetch('/tetgen/tetgen.js');
      if (!response.ok) throw new Error('tetgen.js not found');
      const scriptText = await response.text();
      const script = document.createElement('script');
      script.textContent = scriptText;
      document.head.appendChild(script);
      if ((window as any).TetGen) {
        tetgenModule = (window as any).TetGen;
        tetgenAvailable = true;
        return true;
      }
    } catch (e) {
    }

    tetgenAvailable = false;
    return false;
  } catch (e) {
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
    } catch (e) {
      console.warn('TetGen failed, falling back to voxel method:', e);
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

  const result = tetgenModule.tetrahedralize(
    numVertices,
    surface.vertices,
    numFaces,
    surface.indices,
    {
      quality: qualityRatio,
      volumeConstraint: targetSize * targetSize * targetSize * 0.5,
      preserveBoundary: true,
      switches: 'pq1.414Y',
    }
  );

  const numNodes = result.numPoints;
  const numElements = result.numTetrahedra;
  const nodes = new Float64Array(result.points);
  const elements = new Uint32Array(result.tetrahedra);

  const numFacesOut = result.numTriangles || 0;
  const surfaceFaces = new Uint32Array(result.triangles || new Uint32Array(0));
  const surfaceNormals = new Float32Array(numFacesOut * 3);

  for (let i = 0; i < numFacesOut; i++) {
    const i0 = surfaceFaces[i * 3];
    const i1 = surfaceFaces[i * 3 + 1];
    const i2 = surfaceFaces[i * 3 + 2];
    const ax = nodes[i1 * 3] - nodes[i0 * 3];
    const ay = nodes[i1 * 3 + 1] - nodes[i0 * 3 + 1];
    const az = nodes[i1 * 3 + 2] - nodes[i0 * 3 + 2];
    const bx = nodes[i2 * 3] - nodes[i0 * 3];
    const by = nodes[i2 * 3 + 1] - nodes[i0 * 3 + 1];
    const bz = nodes[i2 * 3 + 2] - nodes[i0 * 3 + 2];
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

  const mesh: TetMesh = {
    nodes,
    elements,
    surfaceFaces,
    surfaceNormals,
    numNodes,
    numElements,
    numSurfaceFaces: numFacesOut,
  };

  const quality = computeMeshQuality(mesh);
  return { mesh, method: 'tetgen', quality };
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
