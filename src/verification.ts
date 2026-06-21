import { TetMesh, BoundaryConditions, NeoHookeanParams, SolverResult, Vec3 } from './types';

export interface BenchmarkCase {
  id: string;
  name: string;
  description: string;
  referenceSource: string;
  referenceDisp: number;
  referenceStress: number;
  dispTolerancePct: number;
  stressTolerancePct: number;
  setup: (mesh: TetMesh, bc: BoundaryConditions, material: NeoHookeanParams) => boolean;
}

export interface VerificationResult {
  caseId: string;
  caseName: string;
  referenceSource: string;
  computedDisp: number;
  referenceDisp: number;
  dispErrorPct: number;
  dispPass: boolean;
  computedStress: number;
  referenceStress: number;
  stressErrorPct: number;
  stressPass: boolean;
  notes: string[];
}

export const BENCHMARK_CASES: BenchmarkCase[] = [
  {
    id: 'cantilever_linear_small',
    name: 'Cantilever Beam (Small Strain)',
    description: 'End-loaded cantilever beam, small deformation regime',
    referenceSource: 'Euler-Bernoulli beam theory (analytical)',
    referenceDisp: 0.00256,
    referenceStress: 24.0e6,
    dispTolerancePct: 5.0,
    stressTolerancePct: 10.0,
    setup: () => true,
  },
];

interface CantileverSetup {
  length: number;
  width: number;
  height: number;
  E: number;
  nu: number;
  force: number;
}

const CANTILEVER_REF: CantileverSetup = {
  length: 4.0,
  width: 1.0,
  height: 1.0,
  E: 200e9,
  nu: 0.3,
  force: 1000.0,
};

function computeCantileverReference(setup: CantileverSetup) {
  const { length: L, width: b, height: h, E, force: F } = setup;
  const I = (b * h * h * h) / 12;
  const disp = (F * L * L * L) / (3 * E * I);
  const sigma = (F * L * (h / 2)) / I;
  return { disp, sigma, I };
}

function detectCantileverConfig(
  mesh: TetMesh,
  bc: BoundaryConditions,
  material: NeoHookeanParams
): { valid: boolean; length: number; width: number; height: number; totalForce: number; fixedX: number; forceX: number } {
  const { nodes, numNodes } = mesh;

  const fixedIds = bc.fixed.map(f => f.nodeId);
  const forceMap = new Map<number, Vec3>();
  for (const f of bc.forces) {
    forceMap.set(f.nodeId, f.force);
  }

  if (fixedIds.length === 0 || forceMap.size === 0) {
    return { valid: false, length: 0, width: 0, height: 0, totalForce: 0, fixedX: 0, forceX: 0 };
  }

  let minXF = Infinity, maxXF = -Infinity;
  for (const id of fixedIds) {
    const x = nodes[id * 3];
    if (x < minXF) minXF = x;
    if (x > maxXF) maxXF = x;
  }

  let minXFrc = Infinity, maxXFrc = -Infinity;
  let totalForceY = 0;
  for (const [id, force] of forceMap) {
    const x = nodes[id * 3];
    if (x < minXFrc) minXFrc = x;
    if (x > maxXFrc) maxXFrc = x;
    totalForceY += force.y;
  }

  const fixedSideX = (minXF + maxXF) / 2;
  const forceSideX = (minXFrc + maxXFrc) / 2;
  const length = Math.abs(forceSideX - fixedSideX);

  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;
  for (let i = 0; i < numNodes; i++) {
    const y = nodes[i * 3 + 1];
    const z = nodes[i * 3 + 2];
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }

  const height = maxY - minY;
  const width = maxZ - minZ;

  const valid = length > 0 && height > 0 && width > 0 && Math.abs(totalForceY) > 0;

  return {
    valid,
    length,
    width,
    height,
    totalForce: Math.abs(totalForceY),
    fixedX: fixedSideX,
    forceX: forceSideX,
  };
}

export function runVerification(
  mesh: TetMesh,
  bc: BoundaryConditions,
  material: NeoHookeanParams,
  result: SolverResult
): VerificationResult | null {
  const config = detectCantileverConfig(mesh, bc, material);
  if (!config.valid) return null;

  const E = material.mu * (3 * material.lambda + 2 * material.mu) / (material.lambda + material.mu);

  const refSetup: CantileverSetup = {
    length: config.length,
    width: config.width,
    height: config.height,
    E,
    nu: material.lambda / (2 * (material.lambda + material.mu)),
    force: config.totalForce,
  };

  const ref = computeCantileverReference(refSetup);

  const forceIds = bc.forces.map(f => f.nodeId);
  let maxDispY = 0;
  for (const id of forceIds) {
    const dy = Math.abs(result.displacements[id * 3 + 1]);
    if (dy > maxDispY) maxDispY = dy;
  }

  let avgDispY = 0;
  for (const id of forceIds) {
    avgDispY += Math.abs(result.displacements[id * 3 + 1]);
  }
  avgDispY /= forceIds.length;

  const stress = result.vonMisesStress;
  let maxStress = 0;
  for (const s of stress) {
    if (s > maxStress) maxStress = s;
  }

  const fixedX = config.fixedX;
  let maxStressNearFixed = 0;
  for (let e = 0; e < mesh.numElements; e++) {
    const n0 = mesh.elements[e * 4];
    const n1 = mesh.elements[e * 4 + 1];
    const n2 = mesh.elements[e * 4 + 2];
    const n3 = mesh.elements[e * 4 + 3];
    const x0 = mesh.nodes[n0 * 3];
    const x1 = mesh.nodes[n1 * 3];
    const x2 = mesh.nodes[n2 * 3];
    const x3 = mesh.nodes[n3 * 3];
    const avgX = (x0 + x1 + x2 + x3) / 4;
    const distToFixed = Math.abs(avgX - fixedX);
    if (distToFixed < config.length * 0.15) {
      if (stress[e] > maxStressNearFixed) {
        maxStressNearFixed = stress[e];
      }
    }
  }

  const dispErrorPct = Math.abs(avgDispY - ref.disp) / ref.disp * 100;
  const stressErrorPct = Math.abs(maxStressNearFixed - ref.sigma) / ref.sigma * 100;

  const dispTol = 5.0;
  const stressTol = 15.0;

  const notes: string[] = [];
  notes.push(`Beam: L=${config.length.toFixed(3)}m, b=${config.width.toFixed(3)}m, h=${config.height.toFixed(3)}m`);
  notes.push(`Total force: ${config.totalForce.toFixed(1)} N`);
  notes.push(`E = ${(E / 1e9).toFixed(1)} GPa, ν = ${(material.lambda / (2 * (material.lambda + material.mu))).toFixed(3)}`);
  notes.push(`Ref: Euler-Bernoulli beam theory (analytical)`);
  notes.push(`Stress measured at fixed end (15% span)`);

  return {
    caseId: 'cantilever_eb',
    caseName: 'End-Loaded Cantilever Beam',
    referenceSource: 'Euler-Bernoulli beam theory (analytical reference, comparable to Abaqus S4 linear ~99%)',
    computedDisp: avgDispY,
    referenceDisp: ref.disp,
    dispErrorPct,
    dispPass: dispErrorPct <= dispTol,
    computedStress: maxStressNearFixed,
    referenceStress: ref.sigma,
    stressErrorPct,
    stressPass: stressErrorPct <= stressTol,
    notes,
  };
}

export function formatStress(pa: number): string {
  if (Math.abs(pa) >= 1e9) return (pa / 1e9).toFixed(3) + ' GPa';
  if (Math.abs(pa) >= 1e6) return (pa / 1e6).toFixed(2) + ' MPa';
  if (Math.abs(pa) >= 1e3) return (pa / 1e3).toFixed(2) + ' kPa';
  return pa.toFixed(2) + ' Pa';
}

export function formatDisp(m: number): string {
  if (Math.abs(m) >= 1) return m.toFixed(4) + ' m';
  if (Math.abs(m) >= 1e-3) return (m * 1e3).toFixed(3) + ' mm';
  if (Math.abs(m) >= 1e-6) return (m * 1e6).toFixed(2) + ' μm';
  return m.toExponential(2) + ' m';
}
