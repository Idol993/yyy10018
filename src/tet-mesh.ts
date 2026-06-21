import { SurfaceMesh, TetMesh, Vec3 } from './types';

const NODE_MERGE_TOLERANCE = 1e-8;
const RAY_EPSILON = 1e-8;

function quantize(v: number): number {
  return Math.round(v / NODE_MERGE_TOLERANCE) * NODE_MERGE_TOLERANCE;
}

function nodeKey(x: number, y: number, z: number): string {
  const qx = quantize(x);
  const qy = quantize(y);
  const qz = quantize(z);
  return `${qx},${qy},${qz}`;
}

function computeAABB(surface: SurfaceMesh): { min: Vec3; max: Vec3 } {
  const verts = surface.vertices;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < verts.length; i += 3) {
    minX = Math.min(minX, verts[i]);
    minY = Math.min(minY, verts[i + 1]);
    minZ = Math.min(minZ, verts[i + 2]);
    maxX = Math.max(maxX, verts[i]);
    maxY = Math.max(maxY, verts[i + 1]);
    maxZ = Math.max(maxZ, verts[i + 2]);
  }
  return {
    min: { x: minX, y: minY, z: minZ },
    max: { x: maxX, y: maxY, z: maxZ }
  };
}

function rayTriangleIntersect(
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  v0x: number, v0y: number, v0z: number,
  v1x: number, v1y: number, v1z: number,
  v2x: number, v2y: number, v2z: number
): number | null {
  const edge1x = v1x - v0x;
  const edge1y = v1y - v0y;
  const edge1z = v1z - v0z;
  const edge2x = v2x - v0x;
  const edge2y = v2y - v0y;
  const edge2z = v2z - v0z;
  const hx = dy * edge2z - dz * edge2y;
  const hy = dz * edge2x - dx * edge2z;
  const hz = dx * edge2y - dy * edge2x;
  const a = edge1x * hx + edge1y * hy + edge1z * hz;
  if (Math.abs(a) < RAY_EPSILON) return null;
  const f = 1 / a;
  const sx = ox - v0x;
  const sy = oy - v0y;
  const sz = oz - v0z;
  const u = f * (sx * hx + sy * hy + sz * hz);
  if (u < 0 || u > 1) return null;
  const qx = sy * edge1z - sz * edge1y;
  const qy = sz * edge1x - sx * edge1z;
  const qz = sx * edge1y - sy * edge1x;
  const v = f * (dx * qx + dy * qy + dz * qz);
  if (v < 0 || u + v > 1) return null;
  const t = f * (edge2x * qx + edge2y * qy + edge2z * qz);
  if (t > RAY_EPSILON) return t;
  return null;
}

function isPointInside(
  px: number, py: number, pz: number,
  surface: SurfaceMesh
): boolean {
  const verts = surface.vertices;
  const indices = surface.indices;
  const dx = 1, dy = 0, dz = 0;
  let count = 0;
  for (let i = 0; i < indices.length; i += 3) {
    const i0 = indices[i] * 3;
    const i1 = indices[i + 1] * 3;
    const i2 = indices[i + 2] * 3;
    const t = rayTriangleIntersect(
      px, py, pz, dx, dy, dz,
      verts[i0], verts[i0 + 1], verts[i0 + 2],
      verts[i1], verts[i1 + 1], verts[i1 + 2],
      verts[i2], verts[i2 + 1], verts[i2 + 2]
    );
    if (t !== null) count++;
  }
  return count % 2 === 1;
}

function raySurfaceIntersection(
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  surface: SurfaceMesh
): { x: number; y: number; z: number; t: number } | null {
  const verts = surface.vertices;
  const indices = surface.indices;
  let minT = Infinity;
  let hitX = 0, hitY = 0, hitZ = 0;
  for (let i = 0; i < indices.length; i += 3) {
    const i0 = indices[i] * 3;
    const i1 = indices[i + 1] * 3;
    const i2 = indices[i + 2] * 3;
    const t = rayTriangleIntersect(
      ox, oy, oz, dx, dy, dz,
      verts[i0], verts[i0 + 1], verts[i0 + 2],
      verts[i1], verts[i1 + 1], verts[i1 + 2],
      verts[i2], verts[i2 + 1], verts[i2 + 2]
    );
    if (t !== null && t < minT) {
      minT = t;
      hitX = ox + dx * t;
      hitY = oy + dy * t;
      hitZ = oz + dz * t;
    }
  }
  if (minT < Infinity) {
    return { x: hitX, y: hitY, z: hitZ, t: minT };
  }
  return null;
}

function snapToSurface(
  px: number, py: number, pz: number,
  surface: SurfaceMesh
): { x: number; y: number; z: number } {
  const verts = surface.vertices;
  const indices = surface.indices;
  let minDist = Infinity;
  let bestX = px, bestY = py, bestZ = pz;
  for (let i = 0; i < indices.length; i += 3) {
    const i0 = indices[i] * 3;
    const i1 = indices[i + 1] * 3;
    const i2 = indices[i + 2] * 3;
    const v0x = verts[i0], v0y = verts[i0 + 1], v0z = verts[i0 + 2];
    const v1x = verts[i1], v1y = verts[i1 + 1], v1z = verts[i1 + 2];
    const v2x = verts[i2], v2y = verts[i2 + 1], v2z = verts[i2 + 2];
    const edge1x = v1x - v0x, edge1y = v1y - v0y, edge1z = v1z - v0z;
    const edge2x = v2x - v0x, edge2y = v2y - v0y, edge2z = v2z - v0z;
    const vpx = px - v0x, vpy = py - v0y, vpz = pz - v0z;
    const dot11 = edge1x * edge1x + edge1y * edge1y + edge1z * edge1z;
    const dot12 = edge1x * edge2x + edge1y * edge2y + edge1z * edge2z;
    const dot22 = edge2x * edge2x + edge2y * edge2y + edge2z * edge2z;
    const dotp1 = vpx * edge1x + vpy * edge1y + vpz * edge1z;
    const dotp2 = vpx * edge2x + vpy * edge2y + vpz * edge2z;
    const denom = dot11 * dot22 - dot12 * dot12;
    let u = (dot22 * dotp1 - dot12 * dotp2) / (denom || 1);
    let v = (dot11 * dotp2 - dot12 * dotp1) / (denom || 1);
    u = Math.max(0, Math.min(1, u));
    v = Math.max(0, Math.min(1, v));
    if (u + v > 1) {
      const s = u + v;
      u /= s;
      v /= s;
    }
    const cx = v0x + u * edge1x + v * edge2x;
    const cy = v0y + u * edge1y + v * edge2y;
    const cz = v0z + u * edge1z + v * edge2z;
    const dx = px - cx, dy = py - cy, dz = pz - cz;
    const dist = dx * dx + dy * dy + dz * dz;
    if (dist < minDist) {
      minDist = dist;
      bestX = cx;
      bestY = cy;
      bestZ = cz;
    }
  }
  return { x: bestX, y: bestY, z: bestZ };
}

function extractSurfaceFaces(elements: Uint32Array, numElements: number): {
  surfaceFaces: Uint32Array;
  numSurfaceFaces: number;
} {
  const faceMap = new Map<string, { count: number; indices: number[] }>();
  const tetFaces = [
    [0, 1, 2],
    [0, 1, 3],
    [0, 2, 3],
    [1, 2, 3]
  ];
  for (let e = 0; e < numElements; e++) {
    const e0 = elements[e * 4];
    const e1 = elements[e * 4 + 1];
    const e2 = elements[e * 4 + 2];
    const e3 = elements[e * 4 + 3];
    for (const face of tetFaces) {
      const a = elements[e * 4 + face[0]];
      const b = elements[e * 4 + face[1]];
      const c = elements[e * 4 + face[2]];
      const sorted = [a, b, c].sort((x, y) => x - y);
      const key = `${sorted[0]},${sorted[1]},${sorted[2]}`;
      const existing = faceMap.get(key);
      if (existing) {
        existing.count++;
      } else {
        faceMap.set(key, { count: 1, indices: [a, b, c] });
      }
    }
  }
  const surfaceFacesList: number[] = [];
  for (const entry of faceMap.values()) {
    if (entry.count === 1) {
      surfaceFacesList.push(entry.indices[0], entry.indices[1], entry.indices[2]);
    }
  }
  const numSurfaceFaces = surfaceFacesList.length / 3;
  const surfaceFaces = new Uint32Array(surfaceFacesList);
  return { surfaceFaces, numSurfaceFaces };
}

function computeSurfaceNormals(
  nodes: Float64Array,
  surfaceFaces: Uint32Array,
  numSurfaceFaces: number
): Float32Array {
  const normals = new Float32Array(numSurfaceFaces * 3);
  for (let f = 0; f < numSurfaceFaces; f++) {
    const i0 = surfaceFaces[f * 3] * 3;
    const i1 = surfaceFaces[f * 3 + 1] * 3;
    const i2 = surfaceFaces[f * 3 + 2] * 3;
    const ax = nodes[i1] - nodes[i0];
    const ay = nodes[i1 + 1] - nodes[i0 + 1];
    const az = nodes[i1 + 2] - nodes[i0 + 2];
    const bx = nodes[i2] - nodes[i0];
    const by = nodes[i2 + 1] - nodes[i0 + 1];
    const bz = nodes[i2 + 2] - nodes[i0 + 2];
    let nx = ay * bz - az * by;
    let ny = az * bx - ax * bz;
    let nz = ax * by - ay * bx;
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 0) {
      nx /= len;
      ny /= len;
      nz /= len;
    }
    normals[f * 3] = nx;
    normals[f * 3 + 1] = ny;
    normals[f * 3 + 2] = nz;
  }
  return normals;
}

export function generateTetMesh(surface: SurfaceMesh, targetSize: number): TetMesh {
  const aabb = computeAABB(surface);
  const margin = targetSize * 0.5;
  const minX = aabb.min.x - margin;
  const minY = aabb.min.y - margin;
  const minZ = aabb.min.z - margin;
  const maxX = aabb.max.x + margin;
  const maxY = aabb.max.y + margin;
  const maxZ = aabb.max.z + margin;

  const nx = Math.max(1, Math.ceil((maxX - minX) / targetSize));
  const ny = Math.max(1, Math.ceil((maxY - minY) / targetSize));
  const nz = Math.max(1, Math.ceil((maxZ - minZ) / targetSize));
  const hx = (maxX - minX) / nx;
  const hy = (maxY - minY) / ny;
  const hz = (maxZ - minZ) / nz;

  const insideVoxels = new Uint8Array(nx * ny * nz);
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const cx = minX + (i + 0.5) * hx;
        const cy = minY + (j + 0.5) * hy;
        const cz = minZ + (k + 0.5) * hz;
        const idx = (k * ny + j) * nx + i;
        insideVoxels[idx] = isPointInside(cx, cy, cz, surface) ? 1 : 0;
      }
    }
  }

  const nodePositions: number[] = [];
  const nodeMap = new Map<string, number>();
  let nodeCount = 0;

  function addNode(x: number, y: number, z: number): number {
    const key = nodeKey(x, y, z);
    let idx = nodeMap.get(key);
    if (idx === undefined) {
      idx = nodeCount++;
      nodeMap.set(key, idx);
      nodePositions.push(x, y, z);
    }
    return idx;
  }

  const cornerOffsets = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
  ];

  const voxelCornerIds = new Uint32Array(8);
  const elementList: number[] = [];

  const cubeSplit = [
    0, 1, 2, 6,
    0, 2, 3, 6,
    0, 3, 7, 6,
    0, 7, 4, 6,
    0, 4, 5, 6,
  ];

  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const voxelIdx = (k * ny + j) * nx + i;
        if (!insideVoxels[voxelIdx]) continue;

        for (let c = 0; c < 8; c++) {
          const [oi, oj, ok] = cornerOffsets[c];
          const cx = minX + (i + oi) * hx;
          const cy = minY + (j + oj) * hy;
          const cz = minZ + (k + ok) * hz;
          const ni = i + oi;
          const nj = j + oj;
          const nk = k + ok;
          const neighborIdx = (nk * ny + nj) * nx + ni;
          const isInsideCorner = ni >= 0 && ni <= nx && nj >= 0 && nj <= ny && nk >= 0 && nk <= nz &&
            ni < nx && nj < ny && nk < nz &&
            insideVoxels[neighborIdx];

          let nodeX = cx, nodeY = cy, nodeZ = cz;
          if (!isInsideCorner) {
            const centerX = minX + (i + 0.5) * hx;
            const centerY = minY + (j + 0.5) * hy;
            const centerZ = minZ + (k + 0.5) * hz;
            const dx = cx - centerX;
            const dy = cy - centerY;
            const dz = cz - centerZ;
            const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (len > 0) {
              const hit = raySurfaceIntersection(centerX, centerY, centerZ, dx / len, dy / len, dz / len, surface);
              if (hit) {
                const snapped = snapToSurface(hit.x, hit.y, hit.z, surface);
                nodeX = snapped.x;
                nodeY = snapped.y;
                nodeZ = snapped.z;
              }
            }
          }
          voxelCornerIds[c] = addNode(nodeX, nodeY, nodeZ);
        }

        for (let t = 0; t < 5; t++) {
          const a = voxelCornerIds[cubeSplit[t * 4]];
          const b = voxelCornerIds[cubeSplit[t * 4 + 1]];
          const c = voxelCornerIds[cubeSplit[t * 4 + 2]];
          const d = voxelCornerIds[cubeSplit[t * 4 + 3]];
          elementList.push(a, b, c, d);
        }
      }
    }
  }

  const numElements = elementList.length / 4;
  const nodes = new Float64Array(nodePositions);
  const elements = new Uint32Array(elementList);
  const { surfaceFaces, numSurfaceFaces } = extractSurfaceFaces(elements, numElements);
  const surfaceNormals = computeSurfaceNormals(nodes, surfaceFaces, numSurfaceFaces);

  return {
    nodes,
    elements,
    surfaceFaces,
    surfaceNormals,
    numNodes: nodeCount,
    numElements,
    numSurfaceFaces
  };
}

export function findNearestNode(nodes: Float64Array, point: Vec3, maxDist: number = Infinity): number {
  let bestIdx = -1;
  let bestDistSq = maxDist * maxDist;
  const numNodes = nodes.length / 3;
  for (let i = 0; i < numNodes; i++) {
    const dx = nodes[i * 3] - point.x;
    const dy = nodes[i * 3 + 1] - point.y;
    const dz = nodes[i * 3 + 2] - point.z;
    const distSq = dx * dx + dy * dy + dz * dz;
    if (distSq < bestDistSq) {
      bestDistSq = distSq;
      bestIdx = i;
    }
  }
  return bestIdx;
}

export function generateSampleCube(size: number = 1): TetMesh {
  return generateBeam(size, size, size, 1);
}

export function generateBeam(
  length: number = 5,
  width: number = 1,
  height: number = 1,
  divs: number = 5
): TetMesh {
  const nx = divs;
  const ny = Math.max(1, Math.ceil(width / (length / divs)));
  const nz = Math.max(1, Math.ceil(height / (length / divs)));
  const hx = length / nx;
  const hy = width / ny;
  const hz = height / nz;

  const nodePositions: number[] = [];
  const nodeMap = new Map<string, number>();
  let nodeCount = 0;

  function addNode(x: number, y: number, z: number): number {
    const key = nodeKey(x, y, z);
    let idx = nodeMap.get(key);
    if (idx === undefined) {
      idx = nodeCount++;
      nodeMap.set(key, idx);
      nodePositions.push(x, y, z);
    }
    return idx;
  }

  const cornerOffsets = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
  ];

  const cubeSplit = [
    0, 1, 2, 6,
    0, 2, 3, 6,
    0, 3, 7, 6,
    0, 7, 4, 6,
    0, 4, 5, 6,
  ];

  const elementList: number[] = [];
  const voxelCornerIds = new Uint32Array(8);

  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        for (let c = 0; c < 8; c++) {
          const [oi, oj, ok] = cornerOffsets[c];
          const cx = (i + oi) * hx - length / 2;
          const cy = (j + oj) * hy - width / 2;
          const cz = (k + ok) * hz - height / 2;
          voxelCornerIds[c] = addNode(cx, cy, cz);
        }
        for (let t = 0; t < 5; t++) {
          const a = voxelCornerIds[cubeSplit[t * 4]];
          const b = voxelCornerIds[cubeSplit[t * 4 + 1]];
          const c = voxelCornerIds[cubeSplit[t * 4 + 2]];
          const d = voxelCornerIds[cubeSplit[t * 4 + 3]];
          elementList.push(a, b, c, d);
        }
      }
    }
  }

  const numElements = elementList.length / 4;
  const nodes = new Float64Array(nodePositions);
  const elements = new Uint32Array(elementList);
  const { surfaceFaces, numSurfaceFaces } = extractSurfaceFaces(elements, numElements);
  const surfaceNormals = computeSurfaceNormals(nodes, surfaceFaces, numSurfaceFaces);

  return {
    nodes,
    elements,
    surfaceFaces,
    surfaceNormals,
    numNodes: nodeCount,
    numElements,
    numSurfaceFaces
  };
}
