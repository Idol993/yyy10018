import { loadModelFile } from './mesh-loader';
import { generateTetMesh, generateSampleCube, generateBeam, findNearestNode } from './tet-mesh';
import { generateHighQualityTetMesh, isTetGenAvailable, computeMeshQuality, type MeshGeneratorType } from './tetgen-wasm';
import { solveNewtonRaphson, computeNeoHookeanPK1, computeVonMisesFromPK1, computeTetShapeGradients, assembleTangentAndInternal, assembleExternalForce } from './fem-solver';
import { WebGPUBackend, pcgSolveCPU } from './gpu-pcg';
import { FEARenderer } from './renderer';
import { ConvergenceChart } from './convergence-chart';
import { InteractionManager } from './interaction';
import { COOBuilder } from './sparse-matrix';
import { runVerification, formatDisp, formatStress, type VerificationResult } from './verification';
import type { TetMesh, BoundaryConditions, NeoHookeanParams, SolverSettings, SolverResult, InteractionMode, Vec3, FixedBC, ForceBC, AppState, CSRSparseMatrix } from './types';

class WebFEAApp {
  private state: AppState;
  private renderer!: FEARenderer;
  private gpuBackend!: WebGPUBackend;
  private interaction!: InteractionManager;
  private convergenceChart!: ConvergenceChart;
  private solverRunning = false;
  private stopRequested = false;

  private readonly els: Record<string, HTMLElement | HTMLInputElement | HTMLCanvasElement | HTMLButtonElement | null> = {};
  private readonly converters = { EtoMuLambda: (E: number, nu: number) => ({ mu: E / (2 * (1 + nu)), lambda: (E * nu) / ((1 + nu) * (1 - 2 * nu)) }) };

  constructor() {
    this.state = {
      mode: 'navigate',
      surfaceMesh: null,
      tetMesh: null,
      meshMethod: 'builtin',
      meshQuality: null,
      bc: { fixed: [], forces: [] },
      material: { mu: 76.9e3, lambda: 115.4e3 },
      settings: { maxIter: 30, tolerance: 1e-6, loadSteps: 5 },
      result: null,
      showWireframe: false,
      showDeformed: true,
      showStress: false,
      deformScale: 1,
    };
  }

  async init(): Promise<void> {
    this.cacheElements();
    this.setupRenderer();
    this.setupGPUBackend();
    this.setupConvergenceChart();
    this.setupInteraction();
    this.setupEventListeners();
    this.syncButtonStates();
    this.loadSampleCube();
    this.updateUI();
  }

  private syncButtonStates(): void {
    (this.els['btn-wireframe'] as HTMLElement).classList.toggle('active', this.state.showWireframe);
    (this.els['btn-deformed'] as HTMLElement).classList.toggle('active', this.state.showDeformed);
    (this.els['btn-stress'] as HTMLElement).classList.toggle('active', this.state.showStress);
  }

  private cacheElements(): void {
    const ids = [
      'viewport', 'btn-import', 'btn-sample', 'btn-mesh', 'btn-fix', 'btn-force',
      'btn-clear-bc', 'btn-solve', 'btn-stop', 'btn-wireframe', 'btn-deformed', 'btn-stress',
      'file-input', 'bc-list', 'convergence-canvas', 'solver-log',
      'info-nodes', 'info-elements', 'info-dofs', 'info-faces', 'info-method', 'info-quality',
      'mat-E', 'mat-nu', 'solve-maxiter', 'solve-tol', 'solve-steps', 'deform-scale',
      'status-iter', 'status-residual', 'status-time',
      'sb-mode', 'sb-gpu', 'sb-mesh', 'sb-solve',
      'stress-legend', 'stress-min', 'stress-max',
      'res-max-disp', 'res-max-stress', 'res-min-stress',
      'ver-case', 'ver-ref', 'ver-disp', 'ver-disp-err', 'ver-stress-err',
      'mode-indicator',
    ];
    for (const id of ids) {
      this.els[id] = document.getElementById(id);
    }
  }

  private setupRenderer(): void {
    const container = this.els['viewport'] as HTMLElement;
    this.renderer = new FEARenderer(container);
  }

  private async setupGPUBackend(): Promise<void> {
    this.gpuBackend = new WebGPUBackend();
    try {
      await this.gpuBackend.init();
      (this.els['sb-gpu'] as HTMLElement).textContent = `GPU: ${this.gpuBackend.getDeviceInfo()}`;
    } catch (e) {
      (this.els['sb-gpu'] as HTMLElement).textContent = 'GPU: WebGPU not available (using CPU)';
    }
  }

  private setupConvergenceChart(): void {
    this.convergenceChart = new ConvergenceChart(this.els['convergence-canvas'] as HTMLCanvasElement);
  }

  private setupInteraction(): void {
    this.interaction = new InteractionManager(this.renderer, null);
    this.interaction.onAddBC = (type: 'fix' | 'force', nodeId: number, force?: Vec3) => {
      if (type === 'fix') {
        const existing = this.state.bc.fixed.find(f => f.nodeId === nodeId);
        if (!existing) {
          this.state.bc.fixed.push({ nodeId, fixedDofs: [true, true, true] });
          this.updateBCList();
          this.renderer.updateBCVisualization(this.state.bc, this.state.tetMesh!);
          this.appendLog(`Fixed constraint added at node ${nodeId}`, '');
        }
      } else if (type === 'force' && force) {
        this.state.bc.forces = this.state.bc.forces.filter(f => f.nodeId !== nodeId);
        this.state.bc.forces.push({ nodeId, force });
        this.updateBCList();
        this.renderer.updateBCVisualization(this.state.bc, this.state.tetMesh!);
        const mag = Math.sqrt(force.x * force.x + force.y * force.y + force.z * force.z).toFixed(2);
        this.appendLog(`Force added at node ${nodeId}: ${mag} N`, '');
      }
    };
  }

  private setupEventListeners(): void {
    (this.els['btn-import'] as HTMLElement).onclick = () => (this.els['file-input'] as HTMLInputElement).click();
    (this.els['file-input'] as HTMLInputElement).onchange = (e) => this.handleFileImport(e);
    (this.els['btn-sample'] as HTMLElement).onclick = () => this.loadSampleCube();
    (this.els['btn-mesh'] as HTMLElement).onclick = () => this.generateMesh();
    (this.els['btn-fix'] as HTMLElement).onclick = () => this.setMode('fix');
    (this.els['btn-force'] as HTMLElement).onclick = () => this.setMode('force');
    (this.els['btn-clear-bc'] as HTMLElement).onclick = () => this.clearBC();
    (this.els['btn-solve'] as HTMLElement).onclick = () => this.runSolver();
    (this.els['btn-stop'] as HTMLElement).onclick = () => this.stopSolver();
    (this.els['btn-wireframe'] as HTMLElement).onclick = () => this.toggleWireframe();
    (this.els['btn-deformed'] as HTMLElement).onclick = () => this.toggleDeformed();
    (this.els['btn-stress'] as HTMLElement).onclick = () => this.toggleStress();

    const materialInputs = ['mat-E', 'mat-nu'] as const;
    for (const id of materialInputs) {
      (this.els[id] as HTMLInputElement).onchange = () => this.updateMaterial();
    }

    const solverInputs = ['solve-maxiter', 'solve-tol', 'solve-steps', 'deform-scale'] as const;
    for (const id of solverInputs) {
      (this.els[id] as HTMLInputElement).onchange = () => this.updateSettings();
    }

    const legend = this.els['stress-legend'] as HTMLElement;
    legend.style.background = 'linear-gradient(to right, #440154, #3b528b, #21918c, #5ec962, #fde725)';
  }

  private async handleFileImport(e: Event): Promise<void> {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file || !this.state.surfaceMesh) return;

    try {
      const surfaceMesh = await loadModelFile(file);
      this.state.surfaceMesh = surfaceMesh;
      this.appendLog(`Loaded surface mesh: ${surfaceMesh.vertices.length / 3} vertices, ${surfaceMesh.indices.length / 3} faces`, '');
    } catch (err) {
      this.appendLog(`Failed to load model: ${err}`, 'err');
    }
    input.value = '';
  }

  private loadSampleCube(): void {
    this.state.tetMesh = generateBeam(4, 1, 1, 4);
    this.state.surfaceMesh = {
      vertices: new Float32Array(this.state.tetMesh.nodes),
      indices: new Uint32Array(this.state.tetMesh.surfaceFaces),
      normals: new Float32Array(this.state.tetMesh.surfaceNormals),
    };
    this.state.meshMethod = 'builtin';
    this.state.meshQuality = computeMeshQuality(this.state.tetMesh);

    this.state.bc = { fixed: [], forces: [] };
    const mesh = this.state.tetMesh;
    for (let i = 0; i < mesh.numNodes; i++) {
      const x = mesh.nodes[i * 3];
      if (x < -1.95) {
        this.state.bc.fixed.push({ nodeId: i, fixedDofs: [true, true, true] });
      }
      if (x > 1.95 && Math.abs(mesh.nodes[i * 3 + 1]) < 0.05 && Math.abs(mesh.nodes[i * 3 + 2]) < 0.05) {
        this.state.bc.forces.push({ nodeId: i, force: { x: 0, y: -100, z: 0 } });
      }
    }

    this.renderer.setMesh(this.state.tetMesh);
    this.renderer.updateBCVisualization(this.state.bc, this.state.tetMesh);
    this.interaction.setMesh(this.state.tetMesh);

    this.appendLog(`Loaded sample beam: ${mesh.numNodes} nodes, ${mesh.numElements} tets`, '');
    this.appendLog(`Mesh method: Built-in structured grid`, '');
    this.appendLog(`Auto-added ${this.state.bc.fixed.length} fixed BCs at x=-2`, '');
    this.appendLog(`Auto-added ${this.state.bc.forces.length} force BCs at x=+2, Fy=-100N`, '');

    this.updateUI();
    this.updateBCList();
  }

  private async generateMesh(): Promise<void> {
    if (!this.state.surfaceMesh) {
      this.appendLog('No surface mesh loaded', 'warn');
      return;
    }
    try {
      const t0 = performance.now();
      const targetSize = parseFloat(prompt('Enter target element size:', '0.3') || '0.3');

      this.appendLog('Generating tetrahedral mesh...', '');
      const hasTetGen = await isTetGenAvailable();

      if (hasTetGen) {
        this.appendLog('Using TetGen WASM for high-quality meshing', '');
      } else {
        this.appendLog('⚠️ TetGen WASM not available, using voxel approximation', 'warn');
      }

      const result = await generateHighQualityTetMesh(this.state.surfaceMesh, targetSize, 1.414);
      this.state.tetMesh = result.mesh;
      this.state.meshMethod = result.method;
      this.state.meshQuality = result.quality;

      const t1 = performance.now();

      this.state.bc = { fixed: [], forces: [] };
      this.renderer.setMesh(this.state.tetMesh);
      this.interaction.setMesh(this.state.tetMesh);

      const methodLabel = result.method === 'tetgen' ? 'TetGen (high quality)' : 'Voxel (approximate)';
      this.appendLog(`Mesh generated: ${this.state.tetMesh.numNodes} nodes, ${this.state.tetMesh.numElements} tets`, 'conv');
      this.appendLog(`Method: ${methodLabel}`, '');
      this.appendLog(`Quality: min=${result.quality.minQuality.toFixed(3)}, avg=${result.quality.avgQuality.toFixed(3)}, maxAR=${result.quality.maxAspectRatio.toFixed(2)}`, '');
      this.appendLog(`Time: ${(t1 - t0).toFixed(1)}ms`, '');
      this.updateUI();
    } catch (err) {
      this.appendLog(`Mesh generation failed: ${err}`, 'err');
    }
  }

  private setMode(mode: InteractionMode): void {
    this.state.mode = mode;
    this.interaction.setMode(mode);

    const btns = ['btn-fix', 'btn-force'] as const;
    for (const id of btns) {
      (this.els[id] as HTMLElement).classList.toggle('active', id === `btn-${mode}`);
    }

    const modeText = mode === 'navigate' ? 'Navigate' : mode === 'fix' ? 'Add Fixed Constraint' : 'Add Force';
    (this.els['sb-mode'] as HTMLElement).textContent = `Mode: ${modeText}`;

    const indicator = this.els['mode-indicator'] as HTMLElement;
    if (mode !== 'navigate') {
      indicator.textContent = mode === 'fix' ? 'Click node to fix' : 'Click & drag to apply force';
      indicator.classList.add('visible');
    } else {
      indicator.classList.remove('visible');
    }
  }

  private clearBC(): void {
    this.state.bc = { fixed: [], forces: [] };
    this.updateBCList();
    this.renderer.updateBCVisualization(this.state.bc, this.state.tetMesh!);
    this.appendLog('All boundary conditions cleared', '');
  }

  private updateMaterial(): void {
    const E = parseFloat((this.els['mat-E'] as HTMLInputElement).value) * 1e3;
    const nu = parseFloat((this.els['mat-nu'] as HTMLInputElement).value);
    this.state.material = this.converters.EtoMuLambda(E, nu);
  }

  private updateSettings(): void {
    this.state.settings.maxIter = parseInt((this.els['solve-maxiter'] as HTMLInputElement).value);
    this.state.settings.tolerance = parseFloat((this.els['solve-tol'] as HTMLInputElement).value);
    this.state.settings.loadSteps = parseInt((this.els['solve-steps'] as HTMLInputElement).value);
    this.state.deformScale = parseFloat((this.els['deform-scale'] as HTMLInputElement).value);
    this.updateDeformation();
  }

  private toggleWireframe(): void {
    this.state.showWireframe = !this.state.showWireframe;
    this.renderer.setWireframe(this.state.showWireframe);
    (this.els['btn-wireframe'] as HTMLElement).classList.toggle('active', this.state.showWireframe);
  }

  private toggleDeformed(): void {
    this.state.showDeformed = !this.state.showDeformed;
    this.renderer.setDeformationVisible(this.state.showDeformed);
    this.updateDeformation();
    (this.els['btn-deformed'] as HTMLElement).classList.toggle('active', this.state.showDeformed);
  }

  private toggleStress(): void {
    this.state.showStress = !this.state.showStress;
    this.renderer.setStressColoring(this.state.showStress);
    this.updateDeformation();
    (this.els['btn-stress'] as HTMLElement).classList.toggle('active', this.state.showStress);
  }

  private async runSolver(): Promise<void> {
    if (!this.state.tetMesh || this.state.bc.fixed.length === 0) {
      this.appendLog('Need mesh and at least one fixed BC', 'warn');
      return;
    }
    if (this.solverRunning) return;

    this.solverRunning = true;
    this.stopRequested = false;
    (this.els['btn-solve'] as HTMLButtonElement).disabled = true;
    (this.els['sb-solve'] as HTMLElement).textContent = 'Solver: Running...';
    (this.els['solver-log'] as HTMLElement).innerHTML = '';
    this.convergenceChart.clear();

    const t0 = performance.now();

    const pcgWrapper = async (K: CSRSparseMatrix, rhs: Float64Array, x0: Float64Array, tol: number, maxIter: number): Promise<Float64Array> => {
      if (this.stopRequested) throw new Error('Solver stopped by user');
      if (this.gpuBackend.isSupported()) {
        return this.gpuBackend.pcgSolve(K, rhs, x0, tol, maxIter);
      }
      return pcgSolveCPU(K, rhs, x0, tol, maxIter);
    };

    const progressCb = (iter: number, residual: number) => {
      (this.els['status-iter'] as HTMLElement).textContent = iter.toString();
      (this.els['status-residual'] as HTMLElement).textContent = residual.toExponential(3);
    };

    try {
      this.appendLog(`Starting Newton-Raphson with ${this.state.settings.loadSteps} load steps`, '');
      this.appendLog(`Mesh: ${this.state.tetMesh.numNodes} nodes, ${this.state.tetMesh.numElements} tets, ${3 * this.state.tetMesh.numNodes} DOFs`, '');

      const result = await solveNewtonRaphson(
        this.state.tetMesh,
        this.state.bc,
        this.state.material,
        this.state.settings,
        pcgWrapper,
        progressCb
      );

      const t1 = performance.now();
      result.solveTimeMs = t1 - t0;
      this.state.result = result;

      for (let i = 0; i < result.convergenceHistory.length; i++) {
        this.appendLog(`Iter ${i + 1}: residual = ${result.convergenceHistory[i].toExponential(4)}`,
          i === result.convergenceHistory.length - 1 ? 'conv' : '');
      }

      if (result.converged) {
        this.appendLog(`Converged in ${result.iterations} iterations, ${result.solveTimeMs.toFixed(1)}ms`, 'conv');
      } else {
        this.appendLog(`Did not converge after ${result.iterations} iterations`, 'warn');
      }

      this.convergenceChart.update(result.convergenceHistory);

      if (!this.state.showStress) {
        this.state.showStress = true;
        this.renderer.setStressColoring(true);
        this.syncButtonStates();
      }

      this.updateDeformation();
      this.updateResultsUI();
      this.runVerification();

      (this.els['sb-solve'] as HTMLElement).textContent = result.converged ? 'Solver: Converged ✓' : 'Solver: Not converged ✗';
      (this.els['status-time'] as HTMLElement).textContent = `${result.solveTimeMs.toFixed(1)} ms`;

    } catch (err) {
      this.appendLog(`Solver error: ${err}`, 'err');
      (this.els['sb-solve'] as HTMLElement).textContent = 'Solver: Error';
    } finally {
      this.solverRunning = false;
      (this.els['btn-solve'] as HTMLButtonElement).disabled = false;
    }
  }

  private stopSolver(): void {
    if (this.solverRunning) {
      this.stopRequested = true;
      this.appendLog('Stop requested...', 'warn');
    }
  }

  private updateDeformation(): void {
    if (!this.state.tetMesh || !this.state.result) return;

    const stress = this.state.showStress ? this.state.result.vonMisesStress : null;
    const scale = this.state.showDeformed ? this.state.deformScale : 0;
    this.renderer.updateDeformation(this.state.result.displacements, scale, stress);
  }

  private updateUI(): void {
    const mesh = this.state.tetMesh;
    if (mesh) {
      (this.els['info-nodes'] as HTMLElement).textContent = mesh.numNodes.toLocaleString();
      (this.els['info-elements'] as HTMLElement).textContent = mesh.numElements.toLocaleString();
      (this.els['info-dofs'] as HTMLElement).textContent = (3 * mesh.numNodes).toLocaleString();
      (this.els['info-faces'] as HTMLElement).textContent = mesh.numSurfaceFaces.toLocaleString();

      const methodLabel = this.state.meshMethod === 'tetgen' ? 'TetGen' :
                         this.state.meshMethod === 'voxel' ? 'Voxel' : 'Built-in';
      (this.els['info-method'] as HTMLElement).textContent = methodLabel;
      (this.els['info-method'] as HTMLElement).style.color =
        this.state.meshMethod === 'tetgen' ? 'var(--accent2)' :
        this.state.meshMethod === 'voxel' ? '#d29922' : '#8b949e';

      if (this.state.meshQuality) {
        const q = this.state.meshQuality.minQuality;
        const qText = q.toFixed(3);
        (this.els['info-quality'] as HTMLElement).textContent = qText;
        (this.els['info-quality'] as HTMLElement).style.color =
          q > 0.5 ? 'var(--accent2)' : q > 0.2 ? '#d29922' : 'var(--danger)';
      } else {
        (this.els['info-quality'] as HTMLElement).textContent = '—';
      }

      const qualityEmoji = this.state.meshQuality
        ? (this.state.meshQuality.minQuality > 0.5 ? ' ✅' : this.state.meshQuality.minQuality > 0.2 ? ' ⚠️' : ' ❌')
        : '';
      (this.els['sb-mesh'] as HTMLElement).textContent =
        `Mesh: ${methodLabel}${qualityEmoji} · ${mesh.numNodes} nodes · ${mesh.numElements} tets`;
    }
  }

  private updateBCList(): void {
    const list = this.els['bc-list'] as HTMLElement;
    list.innerHTML = '';

    for (const bc of this.state.bc.fixed) {
      const div = document.createElement('div');
      div.className = 'bc-item';
      div.innerHTML = `
        <span class="bc-type fixed">FIX</span>
        <span class="bc-node">Node ${bc.nodeId}</span>
        <span class="bc-val">${bc.fixedDofs.map(d => d ? '1' : '0').join(',')}</span>
        <span class="bc-del" data-type="fix" data-id="${bc.nodeId}">✕</span>
      `;
      list.appendChild(div);
    }

    for (const bc of this.state.bc.forces) {
      const mag = Math.sqrt(bc.force.x ** 2 + bc.force.y ** 2 + bc.force.z ** 2).toFixed(1);
      const div = document.createElement('div');
      div.className = 'bc-item';
      div.innerHTML = `
        <span class="bc-type force">FRC</span>
        <span class="bc-node">Node ${bc.nodeId}</span>
        <span class="bc-val">${mag}N</span>
        <span class="bc-del" data-type="force" data-id="${bc.nodeId}">✕</span>
      `;
      list.appendChild(div);
    }

    list.querySelectorAll('.bc-del').forEach(el => {
      el.addEventListener('click', (e) => {
        const type = (e.target as HTMLElement).dataset.type as 'fix' | 'force';
        const id = parseInt((e.target as HTMLElement).dataset.id || '0');
        if (type === 'fix') {
          this.state.bc.fixed = this.state.bc.fixed.filter(f => f.nodeId !== id);
        } else {
          this.state.bc.forces = this.state.bc.forces.filter(f => f.nodeId !== id);
        }
        this.updateBCList();
        this.renderer.updateBCVisualization(this.state.bc, this.state.tetMesh!);
      });
    });
  }

  private updateResultsUI(): void {
    const result = this.state.result;
    if (!result) return;

    const mesh = this.state.tetMesh!;
    let maxDisp = 0;
    for (let i = 0; i < result.displacements.length; i += 3) {
      const d = Math.sqrt(result.displacements[i] ** 2 + result.displacements[i + 1] ** 2 + result.displacements[i + 2] ** 2);
      if (d > maxDisp) maxDisp = d;
    }

    const stress = result.vonMisesStress;
    let maxS = 0, minS = Infinity;
    for (const s of stress) {
      if (s > maxS) maxS = s;
      if (s < minS) minS = s;
    }

    (this.els['res-max-disp'] as HTMLElement).textContent = maxDisp.toExponential(3) + ' m';
    (this.els['res-max-stress'] as HTMLElement).textContent = (maxS / 1e6).toFixed(2) + ' MPa';
    (this.els['res-min-stress'] as HTMLElement).textContent = (minS / 1e6).toFixed(2) + ' MPa';
    (this.els['stress-max'] as HTMLElement).textContent = (maxS / 1e6).toFixed(2) + ' MPa';
    (this.els['stress-min'] as HTMLElement).textContent = (minS / 1e6).toFixed(2) + ' MPa';

    const legend = this.els['stress-legend'] as HTMLElement;
    legend.style.background = 'linear-gradient(to right, #440154, #3b528b, #21918c, #5ec962, #fde725)';
  }

  private runVerification(): void {
    if (!this.state.tetMesh || !this.state.result) {
      (this.els['ver-disp'] as HTMLElement).textContent = '—';
      (this.els['ver-disp-err'] as HTMLElement).textContent = '—';
      (this.els['ver-stress-err'] as HTMLElement).textContent = '—';
      return;
    }

    const ver = runVerification(
      this.state.tetMesh,
      this.state.bc,
      this.state.material,
      this.state.result
    );

    if (!ver) {
      (this.els['ver-disp'] as HTMLElement).textContent = 'N/A';
      (this.els['ver-disp-err'] as HTMLElement).textContent = 'N/A';
      (this.els['ver-stress-err'] as HTMLElement).textContent = 'N/A';
      return;
    }

    (this.els['ver-disp'] as HTMLElement).textContent = formatDisp(ver.referenceDisp);
    (this.els['ver-case'] as HTMLElement).textContent = ver.caseName;
    (this.els['ver-ref'] as HTMLElement).textContent = ver.referenceSource;
    (this.els['ver-disp-err'] as HTMLElement).textContent = ver.dispErrorPct.toFixed(2) + '%';
    (this.els['ver-stress-err'] as HTMLElement).textContent = ver.stressErrorPct.toFixed(2) + '%';

    (this.els['ver-disp-err'] as HTMLElement).style.color =
      ver.dispPass ? 'var(--accent2)' : 'var(--danger)';
    (this.els['ver-stress-err'] as HTMLElement).style.color =
      ver.stressPass ? 'var(--accent2)' : 'var(--danger)';

    (this.els['ver-disp-err'] as HTMLElement).textContent =
      (ver.dispPass ? '✓ ' : '✗ ') + ver.dispErrorPct.toFixed(2) + '%';
    (this.els['ver-stress-err'] as HTMLElement).textContent =
      (ver.stressPass ? '✓ ' : '✗ ') + ver.stressErrorPct.toFixed(2) + '%';

    this.appendLog(`Verification: ${ver.caseName}`, '');
    this.appendLog(`  Ref: ${ver.referenceSource}`, '');
    this.appendLog(`  Disp: ${formatDisp(ver.computedDisp)} vs ref ${formatDisp(ver.referenceDisp)} (${ver.dispErrorPct.toFixed(2)}%) ${ver.dispPass ? '✓' : '✗'}`,
      ver.dispPass ? 'conv' : 'warn');
    this.appendLog(`  Stress: ${formatStress(ver.computedStress)} vs ref ${formatStress(ver.referenceStress)} (${ver.stressErrorPct.toFixed(2)}%) ${ver.stressPass ? '✓' : '✗'}`,
      ver.stressPass ? 'conv' : 'warn');
  }

  private appendLog(msg: string, cls: string = ''): void {
    const log = this.els['solver-log'] as HTMLElement;
    const div = document.createElement('div');
    if (cls) div.className = cls;
    div.textContent = msg;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }
}

const app = new WebFEAApp();
app.init().catch(err => console.error('App init failed:', err));
