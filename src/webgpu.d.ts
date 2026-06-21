interface GPUAdapterInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
}

interface GPUAdapter {
  features: GPUSupportedFeatures;
  limits: GPUSupportedLimits;
  info: GPUAdapterInfo;
  isFallbackAdapter: boolean;
  requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
}

interface GPUDevice {
  adapter: GPUAdapter;
  features: GPUSupportedFeatures;
  limits: GPUSupportedLimits;
  lost: Promise<GPUDeviceLostInfo>;
  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
  createTexture(descriptor: GPUTextureDescriptor): GPUTexture;
  createSampler(descriptor?: GPUSamplerDescriptor): GPUSampler;
  createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout;
  createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout;
  createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
  createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
  createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline;
  createRenderPipeline(descriptor: GPURenderPipelineDescriptor): GPURenderPipeline;
  createCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder;
  createRenderBundleEncoder(descriptor: GPURenderBundleEncoderDescriptor): GPURenderBundleEncoder;
  createQuerySet(descriptor: GPUQuerySetDescriptor): GPUQuerySet;
  queue: GPUQueue;
}

interface GPUBuffer {
  size: number;
  usage: number;
  mapState: GPUMapModeFlags;
  mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

interface GPUTexture {
  width: number;
  height: number;
  depthOrArrayLayers: number;
  mipLevelCount: number;
  sampleCount: number;
  dimension: string;
  format: string;
  usage: number;
  createView(descriptor?: GPUTextureViewDescriptor): GPUTextureView;
  destroy(): void;
}

interface GPUTextureView {
  texture: GPUTexture;
}

interface GPUSampler { }

interface GPUBindGroupLayout { }

interface GPUPipelineLayout { }

interface GPUBindGroup { }

interface GPUShaderModule { }

interface GPUComputePipeline {
  getBindGroupLayout(index: number): GPUBindGroupLayout;
}

interface GPURenderPipeline {
  getBindGroupLayout(index: number): GPUBindGroupLayout;
}

interface GPUCommandEncoder {
  beginComputePass(descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder;
  beginRenderPass(descriptor: GPURenderPassDescriptor): GPURenderPassEncoder;
  copyBufferToBuffer(source: GPUBuffer, sourceOffset: number, destination: GPUBuffer, destinationOffset: number, size: number): void;
  finish(descriptor?: GPUCommandBufferDescriptor): GPUCommandBuffer;
}

interface GPUComputePassEncoder {
  setPipeline(pipeline: GPUComputePipeline): void;
  setBindGroup(index: number, bindGroup: GPUBindGroup): void;
  dispatchWorkgroups(workgroupCountX: number, workgroupCountY?: number, workgroupCountZ?: number): void;
  end(): void;
}

interface GPURenderPassEncoder {
  setPipeline(pipeline: GPURenderPipeline): void;
  setBindGroup(index: number, bindGroup: GPUBindGroup): void;
  setVertexBuffer(slot: number, buffer: GPUBuffer, offset?: number, size?: number): void;
  setIndexBuffer(buffer: GPUBuffer, indexFormat: string, offset?: number, size?: number): void;
  draw(vertexCount: number, instanceCount?: number, firstVertex?: number, firstInstance?: number): void;
  drawIndexed(indexCount: number, instanceCount?: number, firstIndex?: number, baseVertex?: number, firstInstance?: number): void;
  end(): void;
}

interface GPURenderBundleEncoder {
  finish(descriptor?: GPURenderBundleDescriptor): GPURenderBundle;
}

interface GPUCommandBuffer { }

interface GPURenderBundle { }

interface GPUQuerySet { }

interface GPUQueue {
  writeBuffer(buffer: GPUBuffer, bufferOffset: number, data: BufferSource | ArrayBuffer | ArrayBufferView, dataOffset?: number, size?: number): void;
  writeTexture(texture: GPUImageCopyTexture, data: BufferSource, dataLayout: GPUImageDataLayout, size: GPUExtent2D | GPUExtent3D): void;
  copyBufferToBuffer(source: GPUBuffer, sourceOffset: number, destination: GPUBuffer, destinationOffset: number, size: number): void;
  copyTextureToTexture(source: GPUImageCopyTexture, destination: GPUImageCopyTexture, copySize: GPUExtent2D | GPUExtent3D): void;
  copyBufferToTexture(source: GPUImageCopyBuffer, destination: GPUImageCopyTexture, copySize: GPUExtent2D | GPUExtent3D): void;
  copyTextureToBuffer(source: GPUImageCopyTexture, destination: GPUImageCopyBuffer, copySize: GPUExtent2D | GPUExtent3D): void;
  submit(commandBuffers: GPUCommandBuffer[]): void;
  onSubmittedWorkDone(): Promise<void>;
}

type GPUBufferUsageFlags = number;

interface GPUDeviceDescriptor {
  label?: string;
  requiredFeatures?: string[];
  requiredLimits?: Record<string, number>;
}

interface GPUBufferDescriptor {
  label?: string;
  size: number;
  usage: number;
  mappedAtCreation?: boolean;
}

interface GPUTextureDescriptor {
  label?: string;
  size: number | number[];
  mipLevelCount?: number;
  sampleCount?: number;
  dimension?: string;
  format: string;
  usage: number;
}

interface GPUTextureViewDescriptor {
  label?: string;
  format?: string;
  dimension?: string;
  aspect?: string;
  baseMipLevel?: number;
  mipLevelCount?: number;
  baseArrayLayer?: number;
  arrayLayerCount?: number;
}

interface GPUSamplerDescriptor {
  label?: string;
  addressModeU?: string;
  addressModeV?: string;
  addressModeW?: string;
  magFilter?: string;
  minFilter?: string;
  mipmapFilter?: string;
  lodMinClamp?: number;
  lodMaxClamp?: number;
  compare?: string;
  maxAnisotropy?: number;
}

interface GPUBindGroupLayoutDescriptor {
  label?: string;
  entries: Array<{
    binding: number;
    visibility: number;
    buffer?: { type?: string; hasDynamicOffset?: boolean; minBindingSize?: number };
    sampler?: { type?: string };
    texture?: { sampleType?: string; viewDimension?: string; multisampled?: boolean };
    storageTexture?: { access: string; format: string; viewDimension?: string };
  }>;
}

interface GPUPipelineLayoutDescriptor {
  label?: string;
  bindGroupLayouts: GPUBindGroupLayout[];
}

interface GPUBindGroupDescriptor {
  label?: string;
  layout: GPUBindGroupLayout;
  entries: Array<{ binding: number; resource: any }>;
}

interface GPUShaderModuleDescriptor {
  label?: string;
  code: string;
}

interface GPUComputePipelineDescriptor {
  label?: string;
  layout: GPUPipelineLayout | string;
  compute: {
    module: GPUShaderModule;
    entryPoint: string;
    constants?: Record<string, number>;
  };
}

interface GPURenderPipelineDescriptor {
  label?: string;
  layout: GPUPipelineLayout | string;
  vertex: {
    module: GPUShaderModule;
    entryPoint: string;
    buffers?: any[];
    constants?: Record<string, number>;
  };
  primitive?: { topology?: string; stripIndexFormat?: string; frontFace?: string; cullMode?: string };
  depthStencil?: any;
  multisample?: { count?: number; mask?: number; alphaToCoverageEnabled?: boolean };
  fragment?: {
    module: GPUShaderModule;
    entryPoint: string;
    targets: any[];
    constants?: Record<string, number>;
  };
}

interface GPUCommandEncoderDescriptor {
  label?: string;
}

interface GPUComputePassDescriptor {
  label?: string;
  timestampWrites?: any;
}

interface GPURenderPassDescriptor {
  label?: string;
  colorAttachments: any[];
  depthStencilAttachment?: any;
  occlusionQuerySet?: GPUQuerySet;
  timestampWrites?: any;
}

interface GPUCommandBufferDescriptor {
  label?: string;
}

interface GPURenderBundleDescriptor {
  label?: string;
}

interface GPURenderBundleEncoderDescriptor {
  label?: string;
  colorFormats: string[];
  depthStencilFormat?: string;
  sampleCount?: number;
}

interface GPUQuerySetDescriptor {
  label?: string;
  type: string;
  count: number;
}

interface GPUSupportedFeatures {
  has(feature: string): boolean;
  values(): IterableIterator<string>;
  size: number;
}

interface GPUSupportedLimits {
  maxTextureDimension1D: number;
  maxTextureDimension2D: number;
  maxTextureDimension3D: number;
  maxTextureArrayLayers: number;
  maxBindGroups: number;
  maxDynamicUniformBuffersPerPipelineLayout: number;
  maxDynamicStorageBuffersPerPipelineLayout: number;
  maxSampledTexturesPerShaderStage: number;
  maxSamplersPerShaderStage: number;
  maxStorageBuffersPerShaderStage: number;
  maxStorageTexturesPerShaderStage: number;
  maxUniformBuffersPerShaderStage: number;
  maxUniformBufferBindingSize: number;
  maxStorageBufferBindingSize: number;
  minUniformBufferOffsetAlignment: number;
  minStorageBufferOffsetAlignment: number;
  maxVertexBuffers: number;
  maxVertexAttributes: number;
  maxVertexBufferArrayStride: number;
  maxInterStageShaderComponents: number;
  maxComputeWorkgroupStorageSize: number;
  maxComputeInvocationsPerWorkgroup: number;
  maxComputeWorkgroupSizeX: number;
  maxComputeWorkgroupSizeY: number;
  maxComputeWorkgroupSizeZ: number;
  maxComputeWorkgroupsPerDimension: number;
  maxPushConstantSize: number;
  maxColorAttachments: number;
}

interface GPUDeviceLostInfo {
  message: string;
  reason: string;
}

declare const GPUBufferUsage: {
  MAP_READ: number;
  MAP_WRITE: number;
  COPY_SRC: number;
  COPY_DST: number;
  INDEX: number;
  VERTEX: number;
  UNIFORM: number;
  STORAGE: number;
  INDIRECT: number;
  QUERY_RESOLVE: number;
};

declare const GPUShaderStage: {
  VERTEX: number;
  FRAGMENT: number;
  COMPUTE: number;
};

declare const GPUMapMode: {
  READ: number;
  WRITE: number;
};

interface Navigator {
  gpu: {
    requestAdapter(options?: { powerPreference?: string; forceFallbackAdapter?: boolean }): Promise<GPUAdapter | null>;
    wgslLanguageFeatures?: any;
  };
}
