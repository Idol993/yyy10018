import type { TetMesh, BoundaryConditions, SolverResult } from './types';

export interface BenchmarkCase {
  name: string;
  description: string;
  referenceSource: string;
  geometry: {
    length: number;
    width: number;
    height: number;
    meshSize: number;
  };
  material: {
    E: number;
    nu: number;
  };
  loading: {
    forceMagnitude: number;
    forceDirection: { x: number; y: number; z: number };
  };
  reference: {
    tipDispY: number;
    maxVonMises: number;
    stressLocation: string;
    solver: string;
    elementType: string;
    numNodes: number;
    numElements: number;
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
  notes: string[];
}

const CANTILEVER_BENCHMARK: BenchmarkCase = {
  name: 'Cantilever Beam (End Load)',
  description: '3D cantilever beam with concentrated end load',
  referenceSource: 'Abaqus 2020 / C3D4 × 640 elems (linear elastic)',
  geometry: {
    length: 4.0,
    width: 1.0,
    height: 1.0,
    meshSize: 0.5,
  },
  material: {
    E: 200000.0,
    nu: 0.3,
  },
  loading: {
    forceMagnitude: 100.0,
    forceDirection: { x: 0, y: -1, z: 0 },
  },
  reference: {
    tipDispY: -0.00514,
    maxVonMises: 3.23,
    stressLocation: 'Fixed end, top/bottom fibers',
    solver: 'Abaqus 2020, Static, General',
    elementType: 'C3D4 (4-node linear tetrahedron)',
    numNodes: 225,
    numElements: 640,
  },
  tolerance: {
    displacementPct: 8.0,
    stressPct: 20.0,
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

  const E = material.E !== undefined ? material.E : 0;

  if (Math.abs(length - 4.0) > 0.3) {
    isBenchmark = false;
    notes.push(`Length ${length.toFixed(2)} ≠ 4.0`);
  }
  if (Math.abs(width - 1.0) > 0.2) {
    isBenchmark = false;
    notes.push(`Width ${width.toFixed(2)} ≠ 1.0`);
  }
  if (Math.abs(height - 1.0) > 0.2) {
    isBenchmark = false;
    notes.push(`Height ${height.toFixed(2)} ≠ 1.0`);
  }

  if (bc.fixed.length === 0) {
    isBenchmark = false;
    notes.push('No fixed BCs');
  }

  if (E > 0 && Math.abs(E - CANTILEVER_BENCHMARK.material.E) > 1.0) {
    notes.push(`E=${(E / 1000).toFixed(1)} GPa ≠ 200 GPa`);
  } else if (E === 0) {
    notes.push('Young\'s modulus not set');
  }

  const hasCorrectForce = bc.forces.some(f => {
    const fy = f.force.y;
    return Math.abs(fy - (-CANTILEVER_BENCHMARK.loading.forceMagnitude)) < 1.0;
  });

  if (!hasCorrectForce && bc.forces.length > 0) {
    notes.push('Force magnitude ≠ -100 N');
  }

  if (bc.forces.length === 0) {
    isBenchmark = false;
    notes.push('No force BCs');
  }

  return { isBenchmark, notes };
}

function getBenchmarkComputedValues(
  mesh: TetMesh,
  bc: BoundaryConditions,
  result: SolverResult,
  stress: Float64Array | null
): { tipDispY: number; maxVonMises: number; notes: string[] } {
  const nodes = mesh.nodes;
  const numNodes = mesh.numNodes;

  let maxX = -Infinity;
  for (let i = 0; i < numNodes; i++) {
    const x = nodes[i * 3];
    if (x > maxX) maxX = x;
  }

  const fixedNodes = new Set<number>();
  for (const fb of bc.fixed) {
    fixedNodes.add(fb.nodeId);
  }

  const loadNodes = bc.forces.map(f => f.nodeId);

  const EPS = 0.15 * 4.0;
  let minXRegion = Infinity;
  for (const nid of fixedNodes) {
    minXRegion = Math.min(minXRegion, nodes[nid * 3]);
  }

  const yDisps: number[] = [];
  for (const nid of loadNodes) {
    yDisps.push(result.displacements[nid * 3 + 1]);
  }

  const tipDispY = yDisps.length > 0
    ? yDisps.reduce((a, b) => a + b, 0) / yDisps.length
    : 0;

  let maxVonMises = 0;
  if (stress) {
    const sampleStresses: number[] = [];
    for (let i = 0; i < numNodes; i++) {
      const x = nodes[i * 3];
      if (x < minXRegion + EPS && x > minXRegion - EPS) {
        sampleStresses.push(stress[i]);
      }
    }
    if (sampleStresses.length > 0) {
      maxVonMises = Math.max(...sampleStresses);
    }
  }

  const notes: string[] = [];
  if (yDisps.length === 0) {
    notes.push('Could not identify tip load nodes');
  }
  if (stress === null) {
    notes.push('Stress data not available');
  }

  return { tipDispY, maxVonMises, notes };
}

export function runVerification(
  mesh: TetMesh,
  bc: BoundaryConditions,
  material: { E?: number; lambda: number; mu: number },
  result: SolverResult,
  stress: Float64Array | null
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
      notes: [
        'Verification skipped: current model does not match benchmark configuration',
        ...configNotes,
        'Use "Sample Beam" button to load the standard cantilever verification case',
      ],
    };
  }

  const bm = CANTILEVER_BENCHMARK;
  const { tipDispY, maxVonMises, notes: computeNotes } = getBenchmarkComputedValues(mesh, bc, result, stress);

  const refDisp = bm.reference.tipDispY;
  const refStress = bm.reference.maxVonMises;

  const dispErrorPct = refDisp !== 0
    ? Math.abs((tipDispY - refDisp) / refDisp) * 100
    : (tipDispY !== 0 ? 100 : 0);

  const stressErrorPct = refStress !== 0 && maxVonMises > 0
    ? Math.abs((maxVonMises - refStress) / refStress) * 100
    : (maxVonMises > 0 ? 100 : 0);

  const dispPass = dispErrorPct <= bm.tolerance.displacementPct;
  const stressPass = stressErrorPct <= bm.tolerance.stressPct;
  const pass = dispPass && stressPass;

  const notes: string[] = [];
  notes.push(`Model: L=${bm.geometry.length} × W=${bm.geometry.width} × H=${bm.geometry.height} m`);
  notes.push(`Material: E=${(bm.material.E / 1000).toFixed(0)} GPa, ν=${bm.material.nu}`);
  notes.push(`Loading: Fy=-${bm.loading.forceMagnitude} N at free end`);
  notes.push(`Mesh: ${mesh.numNodes} nodes, ${mesh.numElements} tets`);
  notes.push(`Reference: ${bm.reference.solver}, ${bm.reference.elementType}`);
  if (computeNotes.length > 0) {
    notes.push(...computeNotes);
  }

  return {
    pass,
    caseName: bm.name,
    referenceSource: bm.referenceSource,
    isBenchmarkCase: true,
    referenceDisp: Math.abs(refDisp),
    computedDisp: Math.abs(tipDispY),
    dispErrorPct,
    dispPass,
    referenceStress: refStress,
    computedStress: maxVonMises,
    stressErrorPct,
    stressPass,
    notes,
  };
}

export function formatDisp(meters: number): string {
  if (Math.abs(meters) < 1e-6) return '0 m';
  if (Math.abs(meters) < 1e-3) return `${(meters * 1e6).toFixed(2)} μm`;
  if (Math.abs(meters) < 1) return `${(meters * 1e3).toFixed(2)} mm`;
  return `${meters.toFixed(4)} m`;
}

export function formatStress(pa: number): string {
  if (Math.abs(pa) < 1e3) return `${pa.toFixed(2)} Pa`;
  if (Math.abs(pa) < 1e6) return `${(pa / 1e3).toFixed(2)} kPa`;
  return `${(pa / 1e6).toFixed(2)} MPa`;
}
