import { COOBuilder } from './sparse-matrix';
import { TetMesh, BoundaryConditions, NeoHookeanParams, SolverSettings, SolverResult, CSRSparseMatrix } from './types';

function mat3Det(m: Float64Array): number {
  return m[0] * (m[4] * m[8] - m[5] * m[7]) -
         m[3] * (m[1] * m[8] - m[2] * m[7]) +
         m[6] * (m[1] * m[5] - m[2] * m[4]);
}

function mat3Inv(m: Float64Array): Float64Array {
  const det = mat3Det(m);
  const inv = new Float64Array(9);
  inv[0] = (m[4] * m[8] - m[5] * m[7]) / det;
  inv[3] = (m[2] * m[6] - m[0] * m[8]) / det;
  inv[6] = (m[0] * m[5] - m[2] * m[4]) / det;
  inv[1] = (m[5] * m[6] - m[3] * m[8]) / det;
  inv[4] = (m[0] * m[8] - m[2] * m[6]) / det;
  inv[7] = (m[2] * m[3] - m[0] * m[5]) / det;
  inv[2] = (m[3] * m[7] - m[4] * m[6]) / det;
  inv[5] = (m[1] * m[6] - m[0] * m[7]) / det;
  inv[8] = (m[0] * m[4] - m[1] * m[3]) / det;
  return inv;
}

function mat3Transpose(m: Float64Array): Float64Array {
  const t = new Float64Array(9);
  t[0] = m[0]; t[3] = m[1]; t[6] = m[2];
  t[1] = m[3]; t[4] = m[4]; t[7] = m[5];
  t[2] = m[6]; t[5] = m[7]; t[8] = m[8];
  return t;
}

function mat3Mul(a: Float64Array, b: Float64Array): Float64Array {
  const c = new Float64Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      c[i + j * 3] = 0;
      for (let k = 0; k < 3; k++) {
        c[i + j * 3] += a[i + k * 3] * b[k + j * 3];
      }
    }
  }
  return c;
}

export function computeNeoHookeanPK1(F: Float64Array, mu: number, lambda: number): Float64Array {
  const J = mat3Det(F);
  const FT = mat3Transpose(F);
  const C = mat3Mul(FT, F);
  const I1 = C[0] + C[4] + C[8];
  const invFT = mat3Inv(FT);
  const lnJ = Math.log(J);
  const P = new Float64Array(9);
  for (let i = 0; i < 9; i++) {
    P[i] = mu * (F[i] - invFT[i]) + lambda * lnJ * invFT[i];
  }
  return P;
}

export function computeNeoHookeanTangent(F: Float64Array, mu: number, lambda: number): Float64Array {
  const J = mat3Det(F);
  const invF = mat3Inv(F);
  const invFT = mat3Transpose(invF);
  const lnJ = Math.log(J);
  const D = new Float64Array(81);
  
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) {
        for (let l = 0; l < 3; l++) {
          const idx = (i + j * 3) + (k + l * 3) * 9;
          const delta_ij = i === j ? 1 : 0;
          const delta_kl = k === l ? 1 : 0;
          const delta_ik = i === k ? 1 : 0;
          const delta_jl = j === l ? 1 : 0;
          D[idx] = mu * delta_ik * delta_jl +
                   (lambda - mu * lnJ) * invFT[i * 3 + j] * invF[l * 3 + k] +
                   lambda * invF[j * 3 + i] * invFT[k * 3 + l];
        }
      }
    }
  }
  return D;
}

export function computeTetShapeGradients(nodes: Float64Array): { dNdx: Float64Array; detJ: number } {
  const x0 = nodes[0], y0 = nodes[1], z0 = nodes[2];
  const x1 = nodes[3], y1 = nodes[4], z1 = nodes[5];
  const x2 = nodes[6], y2 = nodes[7], z2 = nodes[8];
  const x3 = nodes[9], y3 = nodes[10], z3 = nodes[11];

  const a1 = x1 - x0, b1 = y1 - y0, c1 = z1 - z0;
  const a2 = x2 - x0, b2 = y2 - y0, c2 = z2 - z0;
  const a3 = x3 - x0, b3 = y3 - y0, c3 = z3 - z0;

  const detJ = a1 * (b2 * c3 - b3 * c2) - b1 * (a2 * c3 - a3 * c2) + c1 * (a2 * b3 - a3 * b2);
  const volume = detJ / 6;

  const invDetJ = 1 / detJ;

  const dNdx = new Float64Array(12);

  dNdx[0] = -(b2 * c3 - b3 * c2) * invDetJ;
  dNdx[1] = -(a3 * c2 - a2 * c3) * invDetJ;
  dNdx[2] = -(a2 * b3 - a3 * b2) * invDetJ;

  dNdx[3] = (b3 * c1 - b1 * c3) * invDetJ;
  dNdx[4] = (a1 * c3 - a3 * c1) * invDetJ;
  dNdx[5] = (a3 * b1 - a1 * b3) * invDetJ;

  dNdx[6] = (b1 * c2 - b2 * c1) * invDetJ;
  dNdx[7] = (a2 * c1 - a1 * c2) * invDetJ;
  dNdx[8] = (a1 * b2 - a2 * b1) * invDetJ;

  dNdx[9] = -dNdx[0] - dNdx[3] - dNdx[6];
  dNdx[10] = -dNdx[1] - dNdx[4] - dNdx[7];
  dNdx[11] = -dNdx[2] - dNdx[5] - dNdx[8];

  return { dNdx, detJ: Math.abs(detJ) };
}

export function assembleTangentAndInternal(
  mesh: TetMesh,
  displacements: Float64Array,
  material: NeoHookeanParams,
  builder: COOBuilder,
  internalForce: Float64Array
): void {
  const numElements = mesh.numElements;
  const elements = mesh.elements;
  const nodes = mesh.nodes;

  for (let elemIdx = 0; elemIdx < numElements; elemIdx++) {
    const elemOffset = elemIdx * 4;
    const nodeIds = [
      elements[elemOffset],
      elements[elemOffset + 1],
      elements[elemOffset + 2],
      elements[elemOffset + 3]
    ];

    const elemNodes = new Float64Array(12);
    const elemU = new Float64Array(12);
    for (let i = 0; i < 4; i++) {
      const nid = nodeIds[i];
      const nOff = nid * 3;
      const eOff = i * 3;
      elemNodes[eOff] = nodes[nOff];
      elemNodes[eOff + 1] = nodes[nOff + 1];
      elemNodes[eOff + 2] = nodes[nOff + 2];
      elemU[eOff] = displacements[nOff];
      elemU[eOff + 1] = displacements[nOff + 1];
      elemU[eOff + 2] = displacements[nOff + 2];
    }

    const { dNdx, detJ } = computeTetShapeGradients(elemNodes);
    const volume = detJ / 6;

    const F = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    for (let a = 0; a < 4; a++) {
      const dnOff = a * 3;
      const uOff = a * 3;
      for (let i = 0; i < 3; i++) {
        for (let J = 0; J < 3; J++) {
          F[i + J * 3] += dNdx[dnOff + J] * elemU[uOff + i];
        }
      }
    }

    const P = computeNeoHookeanPK1(F, material.mu, material.lambda);
    const D = computeNeoHookeanTangent(F, material.mu, material.lambda);

    const K_elem = new Float64Array(144);
    const f_int_elem = new Float64Array(12);

    for (let a = 0; a < 4; a++) {
      const dnA = a * 3;
      const rowOff = a * 3;

      for (let i = 0; i < 3; i++) {
        for (let J = 0; J < 3; J++) {
          f_int_elem[rowOff + i] += dNdx[dnA + J] * P[i + J * 3];
        }
      }

      for (let b = 0; b < 4; b++) {
        const dnB = b * 3;
        const colOff = b * 3;

        for (let i = 0; i < 3; i++) {
          for (let j = 0; j < 3; j++) {
            let val = 0;
            for (let I = 0; I < 3; I++) {
              for (let J = 0; J < 3; J++) {
                const dIdx = (i + I * 3) + (j + J * 3) * 9;
                val += dNdx[dnA + I] * D[dIdx] * dNdx[dnB + J];
              }
            }
            K_elem[(rowOff + i) * 12 + (colOff + j)] = val * volume;
          }
        }
      }
    }

    for (let i = 0; i < 12; i++) {
      f_int_elem[i] *= volume;
    }

    for (let a = 0; a < 4; a++) {
      const globalRow = nodeIds[a] * 3;
      const localRow = a * 3;
      for (let i = 0; i < 3; i++) {
        internalForce[globalRow + i] += f_int_elem[localRow + i];
        for (let b = 0; b < 4; b++) {
          const globalCol = nodeIds[b] * 3;
          const localCol = b * 3;
          for (let j = 0; j < 3; j++) {
            builder.add(
              globalRow + i,
              globalCol + j,
              K_elem[(localRow + i) * 12 + (localCol + j)]
            );
          }
        }
      }
    }
  }
}

export function assembleExternalForce(
  mesh: TetMesh,
  bc: BoundaryConditions,
  loadFactor: number
): Float64Array {
  const fExt = new Float64Array(mesh.numNodes * 3);
  for (const forceBC of bc.forces) {
    const nid = forceBC.nodeId;
    const off = nid * 3;
    fExt[off] += forceBC.force.x * loadFactor;
    fExt[off + 1] += forceBC.force.y * loadFactor;
    fExt[off + 2] += forceBC.force.z * loadFactor;
  }
  return fExt;
}

export function computeVonMisesFromPK1(P: Float64Array, F: Float64Array): number {
  const J = mat3Det(F);
  const FT = mat3Transpose(F);
  const sigma = mat3Mul(P, FT);
  const invJ = 1 / J;
  for (let i = 0; i < 9; i++) {
    sigma[i] *= invJ;
  }

  const trace = sigma[0] + sigma[4] + sigma[8];
  const p = trace / 3;

  const s = new Float64Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const idx = i + j * 3;
      s[idx] = sigma[idx] - (i === j ? p : 0);
    }
  }

  let sDotS = 0;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      sDotS += s[i + j * 3] * s[j + i * 3];
    }
  }

  return Math.sqrt(1.5 * sDotS);
}

export async function solveNewtonRaphson(
  mesh: TetMesh,
  bc: BoundaryConditions,
  material: NeoHookeanParams,
  settings: SolverSettings,
  pcgSolver: (K: CSRSparseMatrix, rhs: Float64Array, x0: Float64Array, tol: number, maxIter: number) => Promise<Float64Array>,
  progressCallback?: (iter: number, residual: number) => void
): Promise<SolverResult> {
  const startTime = performance.now();
  const nDofs = mesh.numNodes * 3;
  const u = new Float64Array(nDofs);
  const vonMisesStress = new Float64Array(mesh.numElements);
  const convergenceHistory: number[] = [];
  let totalIterations = 0;
  let converged = true;

  const fixedDofs = new Set<number>();
  for (const fbc of bc.fixed) {
    const nid = fbc.nodeId;
    if (fbc.fixedDofs[0]) fixedDofs.add(nid * 3);
    if (fbc.fixedDofs[1]) fixedDofs.add(nid * 3 + 1);
    if (fbc.fixedDofs[2]) fixedDofs.add(nid * 3 + 2);
  }

  for (let step = 1; step <= settings.loadSteps; step++) {
    const loadFactor = step / settings.loadSteps;
    const fExt = assembleExternalForce(mesh, bc, loadFactor);
    const fExtNorm = Math.sqrt(fExt.reduce((s, v) => s + v * v, 0));
    const tol = settings.tolerance * Math.max(fExtNorm, 1e-10);

    let stepConverged = false;
    for (let iter = 0; iter < settings.maxIter; iter++) {
      totalIterations++;

      const builder = new COOBuilder(nDofs);
      const fInt = new Float64Array(nDofs);
      assembleTangentAndInternal(mesh, u, material, builder, fInt);
      const K = builder.buildCSR();

      const R = new Float64Array(nDofs);
      for (let i = 0; i < nDofs; i++) {
        R[i] = fExt[i] - fInt[i];
      }

      const residual = Math.sqrt(R.reduce((s, v) => s + v * v, 0));
      convergenceHistory.push(residual);

      if (progressCallback) {
        progressCallback(totalIterations, residual);
      }

      if (residual < tol) {
        stepConverged = true;
        break;
      }

      const { mat: K_bc, rhs: R_bc } = applyBCFromSet(K, R, fixedDofs);
      const du = await pcgSolver(K_bc, R_bc, new Float64Array(nDofs), 1e-12, 1000);

      for (let i = 0; i < nDofs; i++) {
        u[i] += du[i];
      }
    }

    if (!stepConverged) {
      converged = false;
      break;
    }
  }

  const elements = mesh.elements;
  const nodes = mesh.nodes;
  for (let elemIdx = 0; elemIdx < mesh.numElements; elemIdx++) {
    const elemOffset = elemIdx * 4;
    const nodeIds = [
      elements[elemOffset],
      elements[elemOffset + 1],
      elements[elemOffset + 2],
      elements[elemOffset + 3]
    ];

    const elemNodes = new Float64Array(12);
    const elemU = new Float64Array(12);
    for (let i = 0; i < 4; i++) {
      const nid = nodeIds[i];
      const nOff = nid * 3;
      const eOff = i * 3;
      elemNodes[eOff] = nodes[nOff];
      elemNodes[eOff + 1] = nodes[nOff + 1];
      elemNodes[eOff + 2] = nodes[nOff + 2];
      elemU[eOff] = u[nOff];
      elemU[eOff + 1] = u[nOff + 1];
      elemU[eOff + 2] = u[nOff + 2];
    }

    const { dNdx } = computeTetShapeGradients(elemNodes);
    const F = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    for (let a = 0; a < 4; a++) {
      const dnOff = a * 3;
      const uOff = a * 3;
      for (let i = 0; i < 3; i++) {
        for (let J = 0; J < 3; J++) {
          F[i + J * 3] += dNdx[dnOff + J] * elemU[uOff + i];
        }
      }
    }

    const P = computeNeoHookeanPK1(F, material.mu, material.lambda);
    vonMisesStress[elemIdx] = computeVonMisesFromPK1(P, F);
  }

  const solveTimeMs = performance.now() - startTime;

  return {
    displacements: u,
    vonMisesStress,
    convergenceHistory,
    iterations: totalIterations,
    converged,
    solveTimeMs
  };
}

function applyBCFromSet(
  mat: CSRSparseMatrix,
  rhs: Float64Array,
  fixedDofs: Set<number>
): { mat: CSRSparseMatrix; rhs: Float64Array } {
  const newRowPtr = new Uint32Array(mat.rowPtr.length);
  newRowPtr.set(mat.rowPtr);

  const newColIdx = new Uint32Array(mat.colIdx.length);
  newColIdx.set(mat.colIdx);

  const newValues = new Float64Array(mat.values.length);
  newValues.set(mat.values);

  const newRhs = new Float64Array(rhs.length);
  newRhs.set(rhs);

  const newDiag = new Float64Array(mat.n);
  newDiag.set(mat.diag);

  for (const dof of fixedDofs) {
    const start = newRowPtr[dof];
    const end = newRowPtr[dof + 1];
    for (let j = start; j < end; j++) {
      newValues[j] = 0;
    }
    for (let i = 0; i < mat.n; i++) {
      const iStart = newRowPtr[i];
      const iEnd = newRowPtr[i + 1];
      for (let j = iStart; j < iEnd; j++) {
        if (newColIdx[j] === dof) {
          newValues[j] = 0;
          break;
        }
      }
    }
    for (let j = start; j < end; j++) {
      if (newColIdx[j] === dof) {
        newValues[j] = 1;
        break;
      }
    }
    newDiag[dof] = 1;
    newRhs[dof] = 0;
  }

  return {
    mat: {
      n: mat.n,
      nnz: mat.nnz,
      rowPtr: newRowPtr,
      colIdx: newColIdx,
      values: newValues,
      diag: newDiag,
    },
    rhs: newRhs,
  };
}
