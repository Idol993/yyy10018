import { SurfaceMesh } from './types';

const VERTEX_MERGE_TOLERANCE = 1e-8;

function quantize(v: number): number {
  return Math.round(v / VERTEX_MERGE_TOLERANCE) * VERTEX_MERGE_TOLERANCE;
}

function vertexKey(x: number, y: number, z: number): string {
  const qx = quantize(x);
  const qy = quantize(y);
  const qz = quantize(z);
  return `${qx},${qy},${qz}`;
}

function computeVertexNormals(
  positions: number[],
  indices: number[],
  vertexCount: number
): Float32Array {
  const normals = new Float32Array(vertexCount * 3);
  for (let i = 0; i < indices.length; i += 3) {
    const i0 = indices[i];
    const i1 = indices[i + 1];
    const i2 = indices[i + 2];
    const ax = positions[i1 * 3] - positions[i0 * 3];
    const ay = positions[i1 * 3 + 1] - positions[i0 * 3 + 1];
    const az = positions[i1 * 3 + 2] - positions[i0 * 3 + 2];
    const bx = positions[i2 * 3] - positions[i0 * 3];
    const by = positions[i2 * 3 + 1] - positions[i0 * 3 + 1];
    const bz = positions[i2 * 3 + 2] - positions[i0 * 3 + 2];
    const nx = ay * bz - az * by;
    const ny = az * bx - ax * bz;
    const nz = ax * by - ay * bx;
    normals[i0 * 3] += nx;
    normals[i0 * 3 + 1] += ny;
    normals[i0 * 3 + 2] += nz;
    normals[i1 * 3] += nx;
    normals[i1 * 3 + 1] += ny;
    normals[i1 * 3 + 2] += nz;
    normals[i2 * 3] += nx;
    normals[i2 * 3 + 1] += ny;
    normals[i2 * 3 + 2] += nz;
  }
  for (let i = 0; i < vertexCount; i++) {
    const ox = normals[i * 3];
    const oy = normals[i * 3 + 1];
    const oz = normals[i * 3 + 2];
    const len = Math.sqrt(ox * ox + oy * oy + oz * oz);
    if (len > 0) {
      normals[i * 3] /= len;
      normals[i * 3 + 1] /= len;
      normals[i * 3 + 2] /= len;
    }
  }
  return normals;
}

export function parseSTL(buffer: ArrayBuffer): SurfaceMesh {
  const bytes = new Uint8Array(buffer);
  let isAscii = false;
  if (bytes.length >= 5) {
    const header = String.fromCharCode(...Array.from(bytes.slice(0, 5)));
    if (header === 'solid') {
      isAscii = true;
      const fullText = new TextDecoder().decode(buffer);
      if (!fullText.includes('facet')) {
        isAscii = false;
      }
    }
  }

  if (isAscii) {
    return parseASCIISTL(new TextDecoder().decode(buffer));
  }
  return parseBinarySTL(buffer);
}

function parseBinarySTL(buffer: ArrayBuffer): SurfaceMesh {
  const view = new DataView(buffer);
  const numFaces = view.getUint32(80, true);
  const positions: number[] = [];
  const faceIndices: number[] = [];
  const vertexMap = new Map<string, number>();
  let vertexCount = 0;
  let offset = 84;
  for (let f = 0; f < numFaces; f++) {
    offset += 12;
    const faceVerts: number[] = [];
    for (let v = 0; v < 3; v++) {
      const x = view.getFloat32(offset, true);
      offset += 4;
      const y = view.getFloat32(offset, true);
      offset += 4;
      const z = view.getFloat32(offset, true);
      offset += 4;
      const key = vertexKey(x, y, z);
      let idx = vertexMap.get(key);
      if (idx === undefined) {
        idx = vertexCount++;
        vertexMap.set(key, idx);
        positions.push(x, y, z);
      }
      faceVerts.push(idx);
    }
    offset += 2;
    faceIndices.push(faceVerts[0], faceVerts[1], faceVerts[2]);
  }
  const vertices = new Float32Array(positions);
  const indices = new Uint32Array(faceIndices);
  const normals = computeVertexNormals(positions, faceIndices, vertexCount);
  return { vertices, indices, normals };
}

function parseASCIISTL(text: string): SurfaceMesh {
  const positions: number[] = [];
  const faceIndices: number[] = [];
  const vertexMap = new Map<string, number>();
  let vertexCount = 0;
  const vertexRegex = /vertex\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)/g;
  let match: RegExpExecArray | null;
  let faceVertCount = 0;
  const currentFace: number[] = [];
  while ((match = vertexRegex.exec(text)) !== null) {
    const x = parseFloat(match[1]);
    const y = parseFloat(match[2]);
    const z = parseFloat(match[3]);
    const key = vertexKey(x, y, z);
    let idx = vertexMap.get(key);
    if (idx === undefined) {
      idx = vertexCount++;
      vertexMap.set(key, idx);
      positions.push(x, y, z);
    }
    currentFace.push(idx);
    faceVertCount++;
    if (faceVertCount === 3) {
      faceIndices.push(currentFace[0], currentFace[1], currentFace[2]);
      currentFace.length = 0;
      faceVertCount = 0;
    }
  }
  const vertices = new Float32Array(positions);
  const indices = new Uint32Array(faceIndices);
  const normals = computeVertexNormals(positions, faceIndices, vertexCount);
  return { vertices, indices, normals };
}

export function parseOBJ(text: string): SurfaceMesh {
  const rawVertices: number[] = [];
  const positions: number[] = [];
  const faceIndices: number[] = [];
  const vertexMap = new Map<string, number>();
  let vertexCount = 0;
  const lines = text.split('\n');
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('v ')) {
      const parts = trimmed.split(/\s+/);
      rawVertices.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]));
    }
  }
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('f ')) continue;
    const parts = trimmed.split(/\s+/);
    const faceVerts: number[] = [];
    for (let i = 1; i < parts.length; i++) {
      const indexStr = parts[i].split('/')[0];
      let idx = parseInt(indexStr, 10);
      if (isNaN(idx)) continue;
      if (idx < 0) {
        idx = rawVertices.length / 3 + idx + 1;
      }
      idx -= 1;
      const x = rawVertices[idx * 3];
      const y = rawVertices[idx * 3 + 1];
      const z = rawVertices[idx * 3 + 2];
      const key = vertexKey(x, y, z);
      let mergedIdx = vertexMap.get(key);
      if (mergedIdx === undefined) {
        mergedIdx = vertexCount++;
        vertexMap.set(key, mergedIdx);
        positions.push(x, y, z);
      }
      faceVerts.push(mergedIdx);
    }
    for (let i = 1; i < faceVerts.length - 1; i++) {
      faceIndices.push(faceVerts[0], faceVerts[i], faceVerts[i + 1]);
    }
  }
  const vertices = new Float32Array(positions);
  const indices = new Uint32Array(faceIndices);
  const normals = computeVertexNormals(positions, faceIndices, vertexCount);
  return { vertices, indices, normals };
}

export async function loadModelFile(file: File): Promise<SurfaceMesh> {
  const name = file.name.toLowerCase();
  if (name.endsWith('.stl')) {
    const buffer = await file.arrayBuffer();
    return parseSTL(buffer);
  }
  if (name.endsWith('.obj')) {
    const text = await file.text();
    return parseOBJ(text);
  }
  throw new Error(`Unsupported file format: ${file.name}`);
}
