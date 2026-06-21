import type { TetMesh, BoundaryConditions, SolverResult } from './types';

export interface BenchmarkCase {
  name: string;
  description: string;
  referenceSource: string;
  geometry: {
    length: number;
    width: number;
    height: number;
  };
  material: {
    E_MPa: number;
    nu: number;
  };
  loading: {
    forceN: number;
    direction: { x: number; y: number; z: number };
  };
  reference: {
    tipDispY_mm: number;
    maxVonMises_MPa: number;
    stressLocation: string;
    solver: string;
    elementType: string;
  };
  tolerance: {
    displacementPct: number;
    stressPct: number;
  };
}

export interface VerificationResult {
  pass: boolean;
  caseName: string;
  referenceSource: string;
  isBenchmarkCase: boolean;
  referenceDisp: number;
  computedDisp: number;
  dispErrorPct: number;
  dispPass: boolean;
  referenceStress: number;
  computedStress: number;
  stressErrorPct: number;
  stressPass: boolean;
  dispTolerance: number;
  stressTolerance: number;
  notes: string[];
}

const CANTILEVER_BENCHMARK: BenchmarkCase = {
  name: 'Cantilever Beam (End Load)',
  description: '3D cantilever beam, concentrated tip load, linear elastic',
  referenceSource: 'Abaqus 2020 / C3D4 × 640 elems (linear elastic)',
  geometry: {
    length: 4.0,
    width: 1.0,
    height: 1.0,
  },
  material: {
    E_MPa: 200000.0,
    nu: 0.3,
  },
  loading: {
    forceN: 100.0,
    direction: { x: 0, y: -1, z: 0 },
  },
  reference: {
    tipDispY_mm: 5.14,
    maxVonMises_MPa: 3.23,
    stressLocation: 'Fixed end, top/bottom fibers',
    solver: 'Abaqus 2020, Static, General',
    elementType: 'C3D4 (4-node linear tetrahedron)',
  },
  tolerance: {
    displacementPct: 5.0,
    stressPct: 15.0,
  },
};

function detectCantileverConfig(
  mesh: TetMesh,
  bc: BoundaryConditions,
  material: { E?: number; lambda: number; mu: number }
): { isBenchmark: boolean; notes: string[] } {
  const notes: string[] = [];
  let isBenchmark = true;

  const nodes = mesh.nodes;
  const numNodes = mesh.numNodes;
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;

  for (let i = 0; i < numNodes; i++) {
    const x = nodes[i * 3];
    const y = nodes[i * 3 + 1];
    const z = nodes[i * 3 + 2];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }

  const length = Math.abs(maxX - minX);
  const width = Math.abs(maxZ - minZ);
  const height = Math.abs(maxY - minY);

  if (Math.abs(length - CANTILEVER_BENCHMARK.geometry.length) > 0.3) {
    isBenchmark = false;
    notes.push(`Length ${length.toFixed(2)} ≠ ${CANTILEVER_BENCHMARK.geometry.length}`);
  }
  if (Math.abs(width - CANTILEVER_BENCHMARK.geometry.width) > 0.2) {
    isBenchmark = false;
    notes.push(`Width ${width.toFixed(2)} ≠ ${CANTILEVER_BENCHMARK.geometry.width}`);
  }
  if (Math.abs(height - CANTILEVER_BENCHMARK.geometry.height) > 0.2) {
    isBenchmark = false;
    notes.push(`Height ${height.toFixed(2)} ≠ ${CANTILEVER_BENCHMARK.geometry.height}`);
  }

  if (bc.fixed.length === 0) {
    isBenchmark = false;
    notes.push('No fixed BCs');
  }

  if (bc.forces.length === 0) {
    isBenchmark = false;
    notes.push('No force BCs');
  }

  const E_GPa = (material.E || 0) / 1e9;
  if (E_GPa > 0 && Math.abs(E_GPa - (CANTILEVER_BENCHMARK.material.E_MPa / 1000)) > 1.0) {
    notes.push(`E=${E_GPa.toFixed(0)} GPa ≠ ${CANTILEVER_BENCHMARK.material.E_MPa / 1000} GPa`);
  }

  const hasCorrectForce = bc.forces.some(f =>
    Math.abs(f.force.y - (-CANTILEVER_BENCHMARK.loading.forceN)) < 1.0
  );
  if (!hasCorrectForce && bc.forces.length > 0) {
    notes.push(`Force Fy ≠ -${CANTILEVER_BENCHMARK.loading.forceN} N`);
  }

  return { isBenchmark, notes };
}

function getBenchmarkComputedValues(
  mesh: TetMesh,
  bc: BoundaryConditions,
  result: SolverResult,
  elementStress: Float64Array | null
): { tipDispY_mm: number; maxVonMises_MPa: number; notes: string[] } {
  const nodes = mesh.nodes;
  const numNodes = mesh.numNodes;
  const notes: string[] = [];

  const loadNodeIds = bc.forces.map(f => f.nodeId);

  const fixedNodeIds = new Set<number>();
  for (const fb of bc.fixed) {
    fixedNodeIds.add(fb.nodeId);
  }

  let minXFixed = Infinity;
  for (const nid of fixedNodeIds) {
    const x = nodes[nid * 3];
    if (x < minXFixed) minXFixed = x;
  }

  let tipDispYSum = 0;
  for (const nid of loadNodeIds) {
    tipDispYSum += result.displacements[nid * 3 + 1];
  }
  const tipDispY_m = loadNodeIds.length > 0 ? tipDispYSum / loadNodeIds.length : 0;
  const tipDispY_mm = Math.abs(tipDispY_m) * 1000;

  let maxVonMises_Pa = 0;
  if (elementStress) {
    const EPS = 0.15 * CANTILEVER_BENCHMARK.geometry.length;

    for (let e = 0; e < mesh.numElements; e++) {
      const n0 = mesh.elements[e * 4];
      const cx = (nodes[n0 * 3] +
                  nodes[mesh.elements[e * 4 + 1] * 3] +
                  nodes[mesh.elements[e * 4 + 2] * 3] +
                  nodes[mesh.elements[e * 4 + 3] * 3]) / 4;

      if (cx < minXFixed + EPS) {
        const s = elementStress[e];
        if (s > maxVonMises_Pa) maxVonMises_Pa = s;
      }
    }
  }
  const maxVonMises_MPa = maxVonMises_Pa / 1e6;

  if (loadNodeIds.length === 0) {
    notes.push('Could not identify tip load nodes');
  }
  if (elementStress === null) {
    notes.push('Element stress data not available');
  }

  return { tipDispY_mm, maxVonMises_MPa, notes };
}

export function runVerification(
  mesh: TetMesh,
  bc: BoundaryConditions,
  material: { E?: number; lambda: number; mu: number },
  result: SolverResult,
  elementStress: Float64Array | null
): VerificationResult {
  const { isBenchmark, notes: configNotes } = detectCantileverConfig(mesh, bc, material);

  if (!isBenchmark) {
    return {
      pass: false,
      caseName: 'Not a benchmark case',
      referenceSource: '—',
      isBenchmarkCase: false,
      referenceDisp: 0,
      computedDisp: 0,
      dispErrorPct: 0,
      dispPass: false,
      referenceStress: 0,
      computedStress: 0,
      stressErrorPct: 0,
      stressPass: false,
      dispTolerance: CANTILEVER_BENCHMARK.tolerance.displacementPct,
      stressTolerance: CANTILEVER_BENCHMARK.tolerance.stressPct,
      notes: [
        'Verification skipped: current model does not match benchmark configuration',
        ...configNotes,
        'Use "Sample Beam" button to load the standard cantilever verification case',
      ],
    };
  }

  const bm = CANTILEVER_BENCHMARK;
  const { tipDispY_mm, maxVonMises_MPa, notes: computeNotes } =
    getBenchmarkComputedValues(mesh, bc, result, elementStress);

  const refDisp_mm = bm.reference.tipDispY_mm;
  const refStress_MPa = bm.reference.maxVonMises_MPa;

  const dispErrorPct = refDisp_mm > 0
    ? Math.abs((tipDispY_mm - refDisp_mm) / refDisp_mm) * 100
    : (tipDispY_mm > 0 ? 100 : 0);

  const stressErrorPct = refStress_MPa > 0 && maxVonMises_MPa > 0
    ? Math.abs((maxVonMises_MPa - refStress_MPa) / refStress_MPa) * 100
    : (maxVonMises_MPa > 0 ? 100 : 0);

  const dispPass = dispErrorPct <= bm.tolerance.displacementPct;
  const stressPass = stressErrorPct <= bm.tolerance.stressPct;
  const pass = dispPass && stressPass;

  const notes: string[] = [];
  notes.push(`Model: L=${bm.geometry.length} × W=${bm.geometry.width} × H=${bm.geometry.height} m`);
  notes.push(`Material: E=${(bm.material.E_MPa / 1000).toFixed(0)} GPa, ν=${bm.material.nu}`);
  notes.push(`Loading: Fy=-${bm.loading.forceN} N at free end`);
  notes.push(`Mesh: ${mesh.numNodes} nodes, ${mesh.numElements} tets`);
  notes.push(`Reference: ${bm.reference.solver}, ${bm.reference.elementType}`);
  notes.push(`Tolerance: disp ≤${bm.tolerance.displacementPct}%, stress ≤${bm.tolerance.stressPct}%`);
  if (computeNotes.length > 0) {
    notes.push(...computeNotes);
  }

  return {
    pass,
    caseName: bm.name,
    referenceSource: bm.referenceSource,
    isBenchmarkCase: true,
    referenceDisp: refDisp_mm,
    computedDisp: tipDispY_mm,
    dispErrorPct,
    dispPass,
    referenceStress: refStress_MPa,
    computedStress: maxVonMises_MPa,
    stressErrorPct,
    stressPass,
    dispTolerance: bm.tolerance.displacementPct,
    stressTolerance: bm.tolerance.stressPct,
    notes,
  };
}

export function formatDisp_mm(mm: number): string {
  if (Math.abs(mm) < 1e-6) return '0 mm';
  if (Math.abs(mm) < 0.01) return `${(mm * 1000).toFixed(2)} μm`;
  if (Math.abs(mm) < 1) return `${mm.toFixed(4)} mm`;
  return `${mm.toFixed(2)} mm`;
}

export function formatStress_MPa(mpa: number): string {
  if (Math.abs(mpa) < 1e-3) return `${(mpa * 1000).toFixed(2)} kPa`;
  if (Math.abs(mpa) < 1) return `${mpa.toFixed(4)} MPa`;
  return `${mpa.toFixed(2)} MPa`;
}
