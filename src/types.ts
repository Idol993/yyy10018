export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface SurfaceMesh {
  vertices: Float32Array;
  indices: Uint32Array;
  normals: Float32Array;
}

export interface TetMesh {
  nodes: Float64Array;
  elements: Uint32Array;
  surfaceFaces: Uint32Array;
  surfaceNormals: Float32Array;
  numNodes: number;
  numElements: number;
  numSurfaceFaces: number;
}

export interface FixedBC {
  nodeId: number;
  fixedDofs: [boolean, boolean, boolean];
}

export interface ForceBC {
  nodeId: number;
  force: Vec3;
}

export interface BoundaryConditions {
  fixed: FixedBC[];
  forces: ForceBC[];
}

export interface NeoHookeanParams {
  mu: number;
  lambda: number;
}

export interface SolverSettings {
  maxIter: number;
  tolerance: number;
  loadSteps: number;
}

export interface SolverResult {
  displacements: Float64Array;
  vonMisesStress: Float64Array;
  convergenceHistory: number[];
  iterations: number;
  converged: boolean;
  solveTimeMs: number;
}

export interface CSRSparseMatrix {
  n: number;
  nnz: number;
  rowPtr: Uint32Array;
  colIdx: Uint32Array;
  values: Float64Array;
  diag: Float64Array;
}

export type InteractionMode = 'navigate' | 'fix' | 'force';

export interface AppState {
  mode: InteractionMode;
  surfaceMesh: SurfaceMesh | null;
  tetMesh: TetMesh | null;
  bc: BoundaryConditions;
  material: NeoHookeanParams;
  settings: SolverSettings;
  result: SolverResult | null;
  showWireframe: boolean;
  showDeformed: boolean;
  showStress: boolean;
  deformScale: number;
}
