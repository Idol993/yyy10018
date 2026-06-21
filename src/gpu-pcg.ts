import { CSRSparseMatrix } from './types';
import { spMV, vecDot, vecNorm, vecAxpy, vecScale } from './sparse-matrix';

const WORKGROUP_SIZE = 64;

const SPMV_SHADER = /* wgsl */ `
struct Params {
  n: u32,
  nnz: u32,
}

@group(0) @binding(0) var<storage, read> rowPtr: array<u32>;
@group(0) @binding(1) var<storage, read> colIdx: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) {
    return;
  }
  let start = rowPtr[i];
  let end = rowPtr[i + 1];
  var sum = 0.0;
  for (var k = start; k < end; k = k + 1) {
    sum = sum + values[k] * x[colIdx[k]];
  }
  y[i] = sum;
}
`;

const AXPY_SHADER = /* wgsl */ `
struct Params {
  n: u32,
  alpha: f32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) {
    return;
  }
  result[i] = params.alpha * x[i] + y[i];
}
`;

const DOT_SHADER = /* wgsl */ `
struct Params {
  n: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> wgSum: array<f32, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x;
  let localIdx = lid.x;
  
  var sum = 0.0;
  var idx = i;
  while (idx < params.n) {
    sum = sum + a[idx] * b[idx];
    idx = idx + ${WORKGROUP_SIZE} * nwg.x;
  }
  
  wgSum[localIdx] = sum;
  workgroupBarrier();
  
  var s = ${WORKGROUP_SIZE} / 2;
  while (s > 0) {
    if (localIdx < u32(s)) {
      wgSum[localIdx] = wgSum[localIdx] + wgSum[localIdx + u32(s)];
    }
    workgroupBarrier();
    s = s / 2;
  }
  
  if (localIdx == 0) {
    partial[gid.x / ${WORKGROUP_SIZE}] = wgSum[0];
  }
}
`;

const SCALE_SHADER = /* wgsl */ `
struct Params {
  n: u32,
  alpha: f32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) {
    return;
  }
  result[i] = params.alpha * x[i];
}
`;

const PRECONDITION_SHADER = /* wgsl */ `
struct Params {
  n: u32,
}

@group(0) @binding(0) var<storage, read> r: array<f32>;
@group(0) @binding(1) var<storage, read> diag: array<f32>;
@group(0) @binding(2) var<storage, read_write> z: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) {
    return;
  }
  z[i] = r[i] / diag[i];
}
`;

const NORM_SHADER = /* wgsl */ `
struct Params {
  n: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> partial: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> wgSum: array<f32, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x;
  let localIdx = lid.x;
  
  var sum = 0.0;
  var idx = i;
  while (idx < params.n) {
    sum = sum + a[idx] * a[idx];
    idx = idx + ${WORKGROUP_SIZE} * nwg.x;
  }
  
  wgSum[localIdx] = sum;
  workgroupBarrier();
  
  var s = ${WORKGROUP_SIZE} / 2;
  while (s > 0) {
    if (localIdx < u32(s)) {
      wgSum[localIdx] = wgSum[localIdx] + wgSum[localIdx + u32(s)];
    }
    workgroupBarrier();
    s = s / 2;
  }
  
  if (localIdx == 0) {
    partial[gid.x / ${WORKGROUP_SIZE}] = wgSum[0];
  }
}
`;

const VEC_ADD_SHADER = /* wgsl */ `
struct Params {
  n: u32,
  beta: f32,
}

@group(0) @binding(0) var<storage, read> z: array<f32>;
@group(0) @binding(1) var<storage, read> p: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) {
    return;
  }
  result[i] = z[i] + params.beta * p[i];
}
`;

export class WebGPUBackend {
  private adapter: GPUAdapter | null = null;
  private device: GPUDevice | null = null;
  private buffers: GPUBuffer[] = [];
  private spmvPipeline: GPUComputePipeline | null = null;
  private axpyPipeline: GPUComputePipeline | null = null;
  private dotPipeline: GPUComputePipeline | null = null;
  private scalePipeline: GPUComputePipeline | null = null;
  private preconditionPipeline: GPUComputePipeline | null = null;
  private normPipeline: GPUComputePipeline | null = null;
  private vecAddPipeline: GPUComputePipeline | null = null;

  async init(): Promise<void> {
    if (!navigator.gpu) {
      throw new Error('WebGPU is not supported');
    }
    this.adapter = await navigator.gpu.requestAdapter();
    if (!this.adapter) {
      throw new Error('No GPU adapter found');
    }
    this.device = await this.adapter.requestDevice();
    this.createPipelines();
  }

  isSupported(): boolean {
    return !!this.device;
  }

  getDeviceInfo(): string {
    if (!this.adapter) return 'Not initialized';
    const info = this.adapter.info;
    return `${info.vendor} - ${info.architecture} - ${info.device}`;
  }

  private createPipelines(): void {
    if (!this.device) return;

    const spmvModule = this.device.createShaderModule({ code: SPMV_SHADER });
    this.spmvPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: spmvModule, entryPoint: 'main' },
    });

    const axpyModule = this.device.createShaderModule({ code: AXPY_SHADER });
    this.axpyPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: axpyModule, entryPoint: 'main' },
    });

    const dotModule = this.device.createShaderModule({ code: DOT_SHADER });
    this.dotPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: dotModule, entryPoint: 'main' },
    });

    const scaleModule = this.device.createShaderModule({ code: SCALE_SHADER });
    this.scalePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: scaleModule, entryPoint: 'main' },
    });

    const preconditionModule = this.device.createShaderModule({ code: PRECONDITION_SHADER });
    this.preconditionPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: preconditionModule, entryPoint: 'main' },
    });

    const normModule = this.device.createShaderModule({ code: NORM_SHADER });
    this.normPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: normModule, entryPoint: 'main' },
    });

    const vecAddModule = this.device.createShaderModule({ code: VEC_ADD_SHADER });
    this.vecAddPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: vecAddModule, entryPoint: 'main' },
    });
  }

  createSparseMatrixBuffers(mat: CSRSparseMatrix): {
    rowPtrBuf: GPUBuffer;
    colIdxBuf: GPUBuffer;
    valuesBuf: GPUBuffer;
    diagBuf: GPUBuffer;
  } {
    if (!this.device) throw new Error('Device not initialized');

    const rowPtrBuf = this.createBuffer(
      mat.rowPtr.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.device.queue.writeBuffer(rowPtrBuf, 0, mat.rowPtr as unknown as ArrayBuffer);

    const colIdxBuf = this.createBuffer(
      mat.colIdx.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.device.queue.writeBuffer(colIdxBuf, 0, mat.colIdx as unknown as ArrayBuffer);

    const valuesF32 = new Float32Array(mat.values);
    const valuesBuf = this.createBuffer(
      valuesF32.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.device.queue.writeBuffer(valuesBuf, 0, valuesF32 as unknown as ArrayBuffer);

    const diagF32 = new Float32Array(mat.diag);
    const diagBuf = this.createBuffer(
      diagF32.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.device.queue.writeBuffer(diagBuf, 0, diagF32 as unknown as ArrayBuffer);

    return { rowPtrBuf, colIdxBuf, valuesBuf, diagBuf };
  }

  createVectorBuffer(size: number, usage?: GPUBufferUsageFlags): GPUBuffer {
    const defaultUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    return this.createBuffer(size * 4, usage ?? defaultUsage);
  }

  writeVectorBuffer(buf: GPUBuffer, data: Float64Array): void {
    if (!this.device) throw new Error('Device not initialized');
    const f32Data = new Float32Array(data);
    this.device.queue.writeBuffer(buf, 0, f32Data as unknown as ArrayBuffer);
  }

  async readVectorBuffer(buf: GPUBuffer, size: number): Promise<Float64Array> {
    if (!this.device) throw new Error('Device not initialized');

    const readBuf = this.device.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buf, 0, readBuf, 0, size * 4);
    this.device.queue.submit([encoder.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ);
    const f32Data = new Float32Array(readBuf.getMappedRange());
    const f64Data = new Float64Array(f32Data);
    readBuf.unmap();
    readBuf.destroy();

    return f64Data;
  }

  private createBuffer(size: number, usage: GPUBufferUsageFlags): GPUBuffer {
    if (!this.device) throw new Error('Device not initialized');
    const buf = this.device.createBuffer({ size, usage });
    this.buffers.push(buf);
    return buf;
  }

  private createUniformBuffer(data: Uint32Array | Float32Array): GPUBuffer {
    if (!this.device) throw new Error('Device not initialized');
    const buf = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buf, 0, data as unknown as ArrayBuffer);
    this.buffers.push(buf);
    return buf;
  }

  private dispatchSpMV(
    rowPtrBuf: GPUBuffer,
    colIdxBuf: GPUBuffer,
    valuesBuf: GPUBuffer,
    xBuf: GPUBuffer,
    yBuf: GPUBuffer,
    n: number,
    nnz: number
  ): void {
    if (!this.device || !this.spmvPipeline) return;

    const params = new Uint32Array([n, nnz]);
    const paramsBuf = this.createUniformBuffer(params);

    const bindGroup = this.device.createBindGroup({
      layout: this.spmvPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: rowPtrBuf } },
        { binding: 1, resource: { buffer: colIdxBuf } },
        { binding: 2, resource: { buffer: valuesBuf } },
        { binding: 3, resource: { buffer: xBuf } },
        { binding: 4, resource: { buffer: yBuf } },
        { binding: 5, resource: { buffer: paramsBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.spmvPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(n / WORKGROUP_SIZE));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  private dispatchAxpy(
    alpha: number,
    xBuf: GPUBuffer,
    yBuf: GPUBuffer,
    resultBuf: GPUBuffer,
    n: number
  ): void {
    if (!this.device || !this.axpyPipeline) return;

    const params = new Float32Array([n, alpha]);
    const paramsBuf = this.createUniformBuffer(params);

    const bindGroup = this.device.createBindGroup({
      layout: this.axpyPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: xBuf } },
        { binding: 1, resource: { buffer: yBuf } },
        { binding: 2, resource: { buffer: resultBuf } },
        { binding: 3, resource: { buffer: paramsBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.axpyPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(n / WORKGROUP_SIZE));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  private async dispatchDot(aBuf: GPUBuffer, bBuf: GPUBuffer, n: number): Promise<number> {
    if (!this.device || !this.dotPipeline) return 0;

    const numWorkgroups = Math.max(1, Math.ceil(n / WORKGROUP_SIZE));
    const partialBuf = this.createVectorBuffer(numWorkgroups, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);

    const params = new Uint32Array([n]);
    const paramsBuf = this.createUniformBuffer(params);

    const bindGroup = this.device.createBindGroup({
      layout: this.dotPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: bBuf } },
        { binding: 2, resource: { buffer: partialBuf } },
        { binding: 3, resource: { buffer: paramsBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.dotPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(numWorkgroups);
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    const partialF64 = await this.readVectorBuffer(partialBuf, numWorkgroups);
    let sum = 0;
    for (let i = 0; i < numWorkgroups; i++) {
      sum += partialF64[i];
    }

    partialBuf.destroy();
    const idx = this.buffers.indexOf(partialBuf);
    if (idx > -1) this.buffers.splice(idx, 1);

    return sum;
  }

  private dispatchScale(
    alpha: number,
    xBuf: GPUBuffer,
    resultBuf: GPUBuffer,
    n: number
  ): void {
    if (!this.device || !this.scalePipeline) return;

    const params = new Float32Array([n, alpha]);
    const paramsBuf = this.createUniformBuffer(params);

    const bindGroup = this.device.createBindGroup({
      layout: this.scalePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: xBuf } },
        { binding: 1, resource: { buffer: resultBuf } },
        { binding: 2, resource: { buffer: paramsBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.scalePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(n / WORKGROUP_SIZE));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  private dispatchPrecondition(
    rBuf: GPUBuffer,
    diagBuf: GPUBuffer,
    zBuf: GPUBuffer,
    n: number
  ): void {
    if (!this.device || !this.preconditionPipeline) return;

    const params = new Uint32Array([n]);
    const paramsBuf = this.createUniformBuffer(params);

    const bindGroup = this.device.createBindGroup({
      layout: this.preconditionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: rBuf } },
        { binding: 1, resource: { buffer: diagBuf } },
        { binding: 2, resource: { buffer: zBuf } },
        { binding: 3, resource: { buffer: paramsBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.preconditionPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(n / WORKGROUP_SIZE));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  private async dispatchNorm(aBuf: GPUBuffer, n: number): Promise<number> {
    if (!this.device || !this.normPipeline) return 0;

    const numWorkgroups = Math.max(1, Math.ceil(n / WORKGROUP_SIZE));
    const partialBuf = this.createVectorBuffer(numWorkgroups, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);

    const params = new Uint32Array([n]);
    const paramsBuf = this.createUniformBuffer(params);

    const bindGroup = this.device.createBindGroup({
      layout: this.normPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: partialBuf } },
        { binding: 2, resource: { buffer: paramsBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.normPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(numWorkgroups);
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    const partialF64 = await this.readVectorBuffer(partialBuf, numWorkgroups);
    let sum = 0;
    for (let i = 0; i < numWorkgroups; i++) {
      sum += partialF64[i];
    }

    partialBuf.destroy();
    const idx = this.buffers.indexOf(partialBuf);
    if (idx > -1) this.buffers.splice(idx, 1);

    return Math.sqrt(sum);
  }

  private dispatchVecAdd(
    zBuf: GPUBuffer,
    pBuf: GPUBuffer,
    beta: number,
    resultBuf: GPUBuffer,
    n: number
  ): void {
    if (!this.device || !this.vecAddPipeline) return;

    const params = new Float32Array([n, beta]);
    const paramsBuf = this.createUniformBuffer(params);

    const bindGroup = this.device.createBindGroup({
      layout: this.vecAddPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: zBuf } },
        { binding: 1, resource: { buffer: pBuf } },
        { binding: 2, resource: { buffer: resultBuf } },
        { binding: 3, resource: { buffer: paramsBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.vecAddPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(n / WORKGROUP_SIZE));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  async pcgSolve(
    mat: CSRSparseMatrix,
    rhs: Float64Array,
    x0: Float64Array,
    tol: number,
    maxIter: number
  ): Promise<Float64Array> {
    if (!this.device) throw new Error('Device not initialized');

    const n = mat.n;
    const nnz = mat.nnz;

    const { rowPtrBuf, colIdxBuf, valuesBuf, diagBuf } = this.createSparseMatrixBuffers(mat);

    const xBuf = this.createVectorBuffer(n);
    const rBuf = this.createVectorBuffer(n);
    const zBuf = this.createVectorBuffer(n);
    const pBuf = this.createVectorBuffer(n);
    const ApBuf = this.createVectorBuffer(n);
    const bBuf = this.createVectorBuffer(n);

    this.writeVectorBuffer(xBuf, x0);
    this.writeVectorBuffer(bBuf, rhs);

    this.dispatchSpMV(rowPtrBuf, colIdxBuf, valuesBuf, xBuf, ApBuf, n, nnz);
    await this.device.queue.onSubmittedWorkDone();

    this.dispatchAxpy(-1, ApBuf, bBuf, rBuf, n);
    await this.device.queue.onSubmittedWorkDone();

    this.dispatchPrecondition(rBuf, diagBuf, zBuf, n);
    await this.device.queue.onSubmittedWorkDone();

    const copyEncoder = this.device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(zBuf, 0, pBuf, 0, n * 4);
    this.device.queue.submit([copyEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    let rsOld = await this.dispatchDot(rBuf, zBuf, n);

    for (let iter = 0; iter < maxIter; iter++) {
      this.dispatchSpMV(rowPtrBuf, colIdxBuf, valuesBuf, pBuf, ApBuf, n, nnz);
      await this.device.queue.onSubmittedWorkDone();

      const pAp = await this.dispatchDot(pBuf, ApBuf, n);
      const alpha = rsOld / pAp;

      this.dispatchAxpy(alpha, pBuf, xBuf, xBuf, n);
      await this.device.queue.onSubmittedWorkDone();

      this.dispatchAxpy(-alpha, ApBuf, rBuf, rBuf, n);
      await this.device.queue.onSubmittedWorkDone();

      const rNorm = await this.dispatchNorm(rBuf, n);
      if (rNorm < tol) {
        break;
      }

      this.dispatchPrecondition(rBuf, diagBuf, zBuf, n);
      await this.device.queue.onSubmittedWorkDone();

      const rsNew = await this.dispatchDot(rBuf, zBuf, n);
      const beta = rsNew / rsOld;

      this.dispatchVecAdd(zBuf, pBuf, beta, pBuf, n);
      await this.device.queue.onSubmittedWorkDone();

      rsOld = rsNew;
    }

    const result = await this.readVectorBuffer(xBuf, n);

    xBuf.destroy();
    rBuf.destroy();
    zBuf.destroy();
    pBuf.destroy();
    ApBuf.destroy();
    bBuf.destroy();

    return result;
  }

  dispose(): void {
    for (const buf of this.buffers) {
      try {
        buf.destroy();
      } catch (e) {
        // ignore
      }
    }
    this.buffers = [];
    this.device = null;
    this.adapter = null;
  }
}

export function pcgSolveCPU(
  mat: CSRSparseMatrix,
  rhs: Float64Array,
  x0: Float64Array,
  tol: number,
  maxIter: number
): Float64Array {
  const n = mat.n;
  let x: Float64Array = new Float64Array(x0);

  const Ax = spMV(mat, x);
  let r: Float64Array = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    r[i] = rhs[i] - Ax[i];
  }

  let z: Float64Array = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    z[i] = r[i] / mat.diag[i];
  }

  let p: Float64Array = new Float64Array(z);
  let rsOld = vecDot(r, z);

  for (let iter = 0; iter < maxIter; iter++) {
    const Ap = spMV(mat, p);
    const pAp = vecDot(p, Ap);
    const alpha = rsOld / pAp;

    x = vecAxpy(alpha, p, x);
    r = vecAxpy(-alpha, Ap, r);

    const rNorm = vecNorm(r);
    if (rNorm < tol) {
      break;
    }

    for (let i = 0; i < n; i++) {
      z[i] = r[i] / mat.diag[i];
    }

    const rsNew = vecDot(r, z);
    const beta = rsNew / rsOld;

    for (let i = 0; i < n; i++) {
      p[i] = z[i] + beta * p[i];
    }

    rsOld = rsNew;
  }

  return x;
}
