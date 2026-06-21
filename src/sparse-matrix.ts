import { CSRSparseMatrix } from './types';

export class COOBuilder {
  private rows: number[] = [];
  private cols: number[] = [];
  private vals: number[] = [];
  private n: number;

  constructor(n: number) {
    this.n = n;
  }

  add(row: number, col: number, value: number): void {
    this.rows.push(row);
    this.cols.push(col);
    this.vals.push(value);
  }

  buildCSR(): CSRSparseMatrix {
    const count = this.rows.length;
    if (count === 0) {
      return {
        n: this.n,
        nnz: 0,
        rowPtr: new Uint32Array(this.n + 1),
        colIdx: new Uint32Array(0),
        values: new Float64Array(0),
        diag: new Float64Array(this.n),
      };
    }

    const entries: Array<{ r: number; c: number; v: number }> = [];
    for (let i = 0; i < count; i++) {
      entries.push({ r: this.rows[i], c: this.cols[i], v: this.vals[i] });
    }

    entries.sort((a, b) => {
      if (a.r !== b.r) return a.r - b.r;
      return a.c - b.c;
    });

    const merged: Array<{ r: number; c: number; v: number }> = [];
    for (let i = 0; i < entries.length; i++) {
      const e = entries[i];
      if (merged.length > 0 && merged[merged.length - 1].r === e.r && merged[merged.length - 1].c === e.c) {
        merged[merged.length - 1].v += e.v;
      } else {
        merged.push({ r: e.r, c: e.c, v: e.v });
      }
    }

    const nnz = merged.length;
    const rowPtr = new Uint32Array(this.n + 1);
    const colIdx = new Uint32Array(nnz);
    const values = new Float64Array(nnz);
    const diag = new Float64Array(this.n);

    for (let i = 0; i < nnz; i++) {
      colIdx[i] = merged[i].c;
      values[i] = merged[i].v;
    }

    let curRow = 0;
    rowPtr[0] = 0;
    for (let i = 0; i < nnz; i++) {
      while (curRow < merged[i].r) {
        curRow++;
        rowPtr[curRow] = i;
      }
    }
    for (let r = curRow + 1; r <= this.n; r++) {
      rowPtr[r] = nnz;
    }

    for (let i = 0; i < this.n; i++) {
      const start = rowPtr[i];
      const end = rowPtr[i + 1];
      for (let j = start; j < end; j++) {
        if (colIdx[j] === i) {
          diag[i] = values[j];
          break;
        }
      }
    }

    return {
      n: this.n,
      nnz,
      rowPtr,
      colIdx,
      values,
      diag,
    };
  }
}

export function spMV(mat: CSRSparseMatrix, x: Float64Array): Float64Array {
  const y = new Float64Array(mat.n);
  for (let i = 0; i < mat.n; i++) {
    let sum = 0;
    const start = mat.rowPtr[i];
    const end = mat.rowPtr[i + 1];
    for (let j = start; j < end; j++) {
      sum += mat.values[j] * x[mat.colIdx[j]];
    }
    y[i] = sum;
  }
  return y;
}

export function vecDot(a: Float64Array, b: Float64Array): number {
  let sum = 0;
  const len = a.length;
  for (let i = 0; i < len; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

export function vecNorm(a: Float64Array): number {
  return Math.sqrt(vecDot(a, a));
}

export function vecAxpy(alpha: number, x: Float64Array, y: Float64Array): Float64Array {
  const result = new Float64Array(y.length);
  for (let i = 0; i < y.length; i++) {
    result[i] = alpha * x[i] + y[i];
  }
  return result;
}

export function vecScale(alpha: number, x: Float64Array): Float64Array {
  const result = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    result[i] = alpha * x[i];
  }
  return result;
}

export function extractDiagonal(mat: CSRSparseMatrix): Float64Array {
  const diag = new Float64Array(mat.n);
  for (let i = 0; i < mat.n; i++) {
    const start = mat.rowPtr[i];
    const end = mat.rowPtr[i + 1];
    for (let j = start; j < end; j++) {
      if (mat.colIdx[j] === i) {
        diag[i] = mat.values[j];
        break;
      }
    }
  }
  return diag;
}

export function applyBC(
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
