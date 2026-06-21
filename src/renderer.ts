import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { TetMesh, Vec3, BoundaryConditions } from './types';

function viridis(t: number): THREE.Color {
  t = Math.max(0, Math.min(1, t));
  const controls = [
    { t: 0.0, r: 0.267, g: 0.004, b: 0.329 },
    { t: 0.25, r: 0.282, g: 0.140, b: 0.458 },
    { t: 0.5, r: 0.253, g: 0.265, b: 0.529 },
    { t: 0.75, r: 0.128, g: 0.566, b: 0.551 },
    { t: 1.0, r: 0.993, g: 0.906, b: 0.144 }
  ];
  for (let i = 0; i < controls.length - 1; i++) {
    if (t <= controls[i + 1].t) {
      const range = controls[i + 1].t - controls[i].t;
      const local = (t - controls[i].t) / range;
      const r = controls[i].r + (controls[i + 1].r - controls[i].r) * local;
      const g = controls[i].g + (controls[i + 1].g - controls[i].g) * local;
      const b = controls[i].b + (controls[i + 1].b - controls[i].b) * local;
      return new THREE.Color(r, g, b);
    }
  }
  return new THREE.Color(controls[controls.length - 1].r, controls[controls.length - 1].g, controls[controls.length - 1].b);
}

export class FEARenderer {
  private container: HTMLElement;
  private scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  private gridHelper: THREE.GridHelper;
  private ambientLight: THREE.AmbientLight;
  private directionalLight: THREE.DirectionalLight;
  private mesh: THREE.Mesh | null = null;
  private wireframe: THREE.LineSegments | null = null;
  private material: THREE.MeshStandardMaterial | null = null;
  private originalPositions: Float32Array | null = null;
  private tetMesh: TetMesh | null = null;
  private bcGroup: THREE.Group | null = null;
  private highlightSphere: THREE.Mesh | null = null;
  private forceArrow: THREE.ArrowHelper | null = null;
  private resizeObserver: ResizeObserver;
  private showWireframe = true;
  private showStressColoring = false;
  private showDeformation = true;
  private animationId: number | null = null;
  private raycaster: THREE.Raycaster;
  private mouse: THREE.Vector2;
  canvas: HTMLCanvasElement;

  constructor(container: HTMLElement) {
    this.container = container;
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1a1a2e);

    const width = container.clientWidth;
    const height = container.clientHeight;

    this.camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    this.camera.position.set(3, 2, 5);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    container.appendChild(this.renderer.domElement);
    this.canvas = this.renderer.domElement;

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;

    this.ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(this.ambientLight);

    this.directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    this.directionalLight.position.set(1, 1, 1);
    this.directionalLight.castShadow = true;
    this.scene.add(this.directionalLight);

    this.gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
    this.scene.add(this.gridHelper);

    this.bcGroup = new THREE.Group();
    this.scene.add(this.bcGroup);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);

    this.render();
  }

  setMesh(mesh: TetMesh): void {
    this.tetMesh = mesh;
    this.clearMesh();

    const geometry = new THREE.BufferGeometry();
    const numNodes = mesh.numNodes;
    const positions = new Float32Array(numNodes * 3);

    for (let i = 0; i < numNodes; i++) {
      positions[i * 3] = mesh.nodes[i * 3];
      positions[i * 3 + 1] = mesh.nodes[i * 3 + 1];
      positions[i * 3 + 2] = mesh.nodes[i * 3 + 2];
    }

    this.originalPositions = new Float32Array(positions);

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setIndex(new THREE.Uint32BufferAttribute(mesh.surfaceFaces, 1));
    geometry.computeVertexNormals();

    this.material = new THREE.MeshStandardMaterial({
      color: 0x4da6ff,
      metalness: 0.1,
      roughness: 0.5,
      side: THREE.DoubleSide
    });

    this.mesh = new THREE.Mesh(geometry, this.material);
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    this.scene.add(this.mesh);

    const wireframeGeometry = new THREE.WireframeGeometry(geometry);
    const wireframeMaterial = new THREE.LineBasicMaterial({ color: 0x000000, opacity: 0.3, transparent: true });
    this.wireframe = new THREE.LineSegments(wireframeGeometry, wireframeMaterial);
    this.scene.add(this.wireframe);

    this.wireframe.visible = this.showWireframe;

    geometry.computeBoundingSphere();
    if (geometry.boundingSphere) {
      const center = geometry.boundingSphere.center;
      const radius = geometry.boundingSphere.radius;
      this.camera.position.set(
        center.x + radius * 2,
        center.y + radius * 1.5,
        center.z + radius * 2.5
      );
      this.controls.target.copy(center);
      this.controls.update();
    }
  }

  private clearMesh(): void {
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      (this.mesh.material as THREE.Material).dispose();
      this.mesh = null;
    }
    if (this.wireframe) {
      this.scene.remove(this.wireframe);
      this.wireframe.geometry.dispose();
      (this.wireframe.material as THREE.Material).dispose();
      this.wireframe = null;
    }
    this.material = null;
    this.originalPositions = null;
  }

  updateDeformation(displacements: Float64Array, scale: number, vonMisesStress: Float64Array | null): void {
    if (!this.mesh || !this.originalPositions || !this.tetMesh) return;

    const geometry = this.mesh.geometry;
    const positionAttr = geometry.getAttribute('position') as THREE.BufferAttribute;
    const positions = positionAttr.array as Float32Array;

    if (this.showDeformation) {
      for (let i = 0; i < this.originalPositions.length; i++) {
        positions[i] = this.originalPositions[i] + displacements[i] * scale;
      }
    } else {
      positions.set(this.originalPositions);
    }

    positionAttr.needsUpdate = true;

    if (vonMisesStress && this.showStressColoring) {
      const numNodes = this.tetMesh.numNodes;
      const elementCount = vonMisesStress.length;
      const nodeStress = new Float64Array(numNodes);
      const nodeCount = new Uint32Array(numNodes);
      const elements = this.tetMesh.elements;

      for (let e = 0; e < elementCount; e++) {
        const stress = vonMisesStress[e];
        for (let n = 0; n < 4; n++) {
          const nodeId = elements[e * 4 + n];
          nodeStress[nodeId] += stress;
          nodeCount[nodeId]++;
        }
      }

      let maxStress = 0;
      for (let i = 0; i < numNodes; i++) {
        if (nodeCount[i] > 0) {
          nodeStress[i] /= nodeCount[i];
          if (nodeStress[i] > maxStress) maxStress = nodeStress[i];
        }
      }

      const colors = new Float32Array(numNodes * 3);
      for (let i = 0; i < numNodes; i++) {
        const stress = nodeStress[i];
        const t = maxStress > 0 ? stress / maxStress : 0;
        const color = viridis(t);
        colors[i * 3] = color.r;
        colors[i * 3 + 1] = color.g;
        colors[i * 3 + 2] = color.b;
      }

      geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
      if (this.material) {
        this.material.vertexColors = true;
        this.material.needsUpdate = true;
      }
    } else {
      geometry.deleteAttribute('color');
      if (this.material) {
        this.material.vertexColors = false;
        this.material.color.setHex(0x4da6ff);
        this.material.needsUpdate = true;
      }
    }

    geometry.computeVertexNormals();
    (geometry.getAttribute('normal') as THREE.BufferAttribute).needsUpdate = true;

    if (this.wireframe) {
      this.wireframe.geometry.dispose();
      this.wireframe.geometry = new THREE.WireframeGeometry(geometry);
    }
  }

  updateBCVisualization(bc: BoundaryConditions, mesh: TetMesh): void {
    if (!this.bcGroup) return;

    this.bcGroup.clear();

    const sphereGeometry = new THREE.SphereGeometry(0.03, 16, 16);
    const redMaterial = new THREE.MeshBasicMaterial({ color: 0xff3333 });
    const blueMaterial = new THREE.MeshBasicMaterial({ color: 0x3333ff });

    for (const fixed of bc.fixed) {
      const sphere = new THREE.Mesh(sphereGeometry, redMaterial);
      const ni = fixed.nodeId * 3;
      sphere.position.set(
        mesh.nodes[ni],
        mesh.nodes[ni + 1],
        mesh.nodes[ni + 2]
      );
      this.bcGroup.add(sphere);
    }

    let maxForceMag = 0;
    for (const force of bc.forces) {
      const mag = Math.sqrt(
        force.force.x * force.force.x +
        force.force.y * force.force.y +
        force.force.z * force.force.z
      );
      if (mag > maxForceMag) maxForceMag = mag;
    }

    for (const force of bc.forces) {
      const ni = force.nodeId * 3;
      const origin = new THREE.Vector3(
        mesh.nodes[ni],
        mesh.nodes[ni + 1],
        mesh.nodes[ni + 2]
      );

      const sphere = new THREE.Mesh(sphereGeometry, blueMaterial);
      sphere.position.copy(origin);
      this.bcGroup.add(sphere);

      const dir = new THREE.Vector3(force.force.x, force.force.y, force.force.z);
      const mag = dir.length();
      if (mag > 0 && maxForceMag > 0) {
        dir.normalize();
        const arrowLength = 0.2 + (mag / maxForceMag) * 0.3;
        const arrowHelper = new THREE.ArrowHelper(
          dir,
          origin,
          arrowLength,
          0x3333ff,
          0.06,
          0.04
        );
        this.bcGroup.add(arrowHelper);
      }
    }
  }

  highlightNode(nodeId: number): void {
    this.clearHighlight();
    if (!this.tetMesh) return;

    const ni = nodeId * 3;
    const geometry = new THREE.SphereGeometry(0.05, 16, 16);
    const material = new THREE.MeshBasicMaterial({ color: 0xffff00 });
    this.highlightSphere = new THREE.Mesh(geometry, material);
    this.highlightSphere.position.set(
      this.tetMesh.nodes[ni],
      this.tetMesh.nodes[ni + 1],
      this.tetMesh.nodes[ni + 2]
    );
    this.scene.add(this.highlightSphere);
  }

  clearHighlight(): void {
    if (this.highlightSphere) {
      this.scene.remove(this.highlightSphere);
      this.highlightSphere.geometry.dispose();
      (this.highlightSphere.material as THREE.Material).dispose();
      this.highlightSphere = null;
    }
  }

  showForceArrow(start: THREE.Vector3, end: THREE.Vector3): void {
    this.hideForceArrow();
    const dir = new THREE.Vector3().subVectors(end, start);
    const length = dir.length();
    if (length > 0) {
      dir.normalize();
      this.forceArrow = new THREE.ArrowHelper(
        dir,
        start.clone(),
        length,
        0x00ff00,
        0.08,
        0.05
      );
      this.scene.add(this.forceArrow);
    }
  }

  hideForceArrow(): void {
    if (this.forceArrow) {
      this.scene.remove(this.forceArrow);
      this.forceArrow.dispose();
      this.forceArrow = null;
    }
  }

  setWireframe(show: boolean): void {
    this.showWireframe = show;
    if (this.wireframe) {
      this.wireframe.visible = show;
    }
  }

  setStressColoring(show: boolean): void {
    this.showStressColoring = show;
  }

  setDeformationVisible(show: boolean): void {
    this.showDeformation = show;
  }

  getRaycasterIntersect(event: MouseEvent): { nodeId: number; point: Vec3 } | null {
    if (!this.mesh || !this.tetMesh) return null;

    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObject(this.mesh);

    if (intersects.length === 0) return null;

    const point = intersects[0].point;

    let nearestId = 0;
    let minDistSq = Infinity;

    for (let i = 0; i < this.tetMesh.numNodes; i++) {
      const ni = i * 3;
      const dx = this.tetMesh.nodes[ni] - point.x;
      const dy = this.tetMesh.nodes[ni + 1] - point.y;
      const dz = this.tetMesh.nodes[ni + 2] - point.z;
      const distSq = dx * dx + dy * dy + dz * dz;
      if (distSq < minDistSq) {
        minDistSq = distSq;
        nearestId = i;
      }
    }

    return {
      nodeId: nearestId,
      point: { x: point.x, y: point.y, z: point.z }
    };
  }

  on(event: string, callback: Function): void {
    this.renderer.domElement.addEventListener(event, callback as EventListener);
  }

  resize(): void {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  render(): void {
    const animate = () => {
      this.animationId = requestAnimationFrame(animate);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    animate();
  }

  dispose(): void {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
    this.resizeObserver.disconnect();
    this.clearHighlight();
    this.hideForceArrow();
    this.clearMesh();
    this.renderer.dispose();
    this.controls.dispose();
    if (this.container.contains(this.renderer.domElement)) {
      this.container.removeChild(this.renderer.domElement);
    }
  }
}
