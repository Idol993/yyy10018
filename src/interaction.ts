import * as THREE from 'three';
import { TetMesh, Vec3, InteractionMode } from './types';
import { FEARenderer } from './renderer';

interface AddBCCallback {
  (type: 'fix', nodeId: number): void;
  (type: 'force', nodeId: number, force: Vec3): void;
}

interface DragState {
  active: boolean;
  startPoint: THREE.Vector3;
  currentPoint: THREE.Vector3;
  nodeId: number;
  plane: THREE.Plane;
}

export class InteractionManager {
  private renderer: FEARenderer;
  private mesh: TetMesh | null;
  private mode: InteractionMode;
  private canvas: HTMLCanvasElement;
  private raycaster: THREE.Raycaster;
  private dragState: DragState;
  onAddBC: AddBCCallback = () => {};

  constructor(renderer: FEARenderer, mesh: TetMesh | null) {
    this.renderer = renderer;
    this.mesh = mesh;
    this.mode = 'navigate';
    this.canvas = renderer.canvas;
    this.raycaster = new THREE.Raycaster();
    this.dragState = {
      active: false,
      startPoint: new THREE.Vector3(),
      currentPoint: new THREE.Vector3(),
      nodeId: -1,
      plane: new THREE.Plane(),
    };
    this.bindEvents();
  }

  setMode(mode: InteractionMode): void {
    this.mode = mode;
    this.renderer.controls.enabled = mode === 'navigate';
    this.canvas.style.cursor = mode === 'navigate' ? 'grab' : 'crosshair';
    this.cancelDrag();
  }

  setMesh(mesh: TetMesh | null): void {
    this.mesh = mesh;
    this.cancelDrag();
  }

  private bindEvents(): void {
    this.canvas.addEventListener('mousedown', this.onMouseDown);
    this.canvas.addEventListener('mousemove', this.onMouseMove);
    this.canvas.addEventListener('mouseup', this.onMouseUp);
    this.canvas.addEventListener('mouseleave', this.onMouseUp);
  }

  private getNormalizedMouse(e: MouseEvent): { x: number; y: number } {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * 2 - 1,
      y: -((e.clientY - rect.top) / rect.height) * 2 + 1,
    };
  }

  private getNearestNode(screenX: number, screenY: number): { nodeId: number; point: THREE.Vector3 } | null {
    if (!this.mesh || this.mesh.numNodes === 0) return null;

    const mouse = this.getNormalizedMouse({ clientX: screenX, clientY: screenY } as MouseEvent);
    this.raycaster.setFromCamera(new THREE.Vector2(mouse.x, mouse.y), this.renderer.camera);

    const ray = this.raycaster.ray;
    let minDist = Infinity;
    let nearestNodeId = -1;
    let nearestPoint = new THREE.Vector3();
    const searchRadius = this.getSearchRadius();

    for (let i = 0; i < this.mesh.numNodes; i++) {
      const px = this.mesh.nodes[i * 3];
      const py = this.mesh.nodes[i * 3 + 1];
      const pz = this.mesh.nodes[i * 3 + 2];
      const nodePos = new THREE.Vector3(px, py, pz);
      const dist = ray.distanceToPoint(nodePos);
      if (dist < minDist && dist < searchRadius) {
        minDist = dist;
        nearestNodeId = i;
        nearestPoint = nodePos;
      }
    }

    return nearestNodeId >= 0 ? { nodeId: nearestNodeId, point: nearestPoint } : null;
  }

  private getSearchRadius(): number {
    const camera = this.renderer.camera;
    if (!this.mesh || this.mesh.numNodes === 0) return 0.1;

    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < this.mesh.numNodes; i++) {
      minX = Math.min(minX, this.mesh.nodes[i * 3]);
      minY = Math.min(minY, this.mesh.nodes[i * 3 + 1]);
      minZ = Math.min(minZ, this.mesh.nodes[i * 3 + 2]);
      maxX = Math.max(maxX, this.mesh.nodes[i * 3]);
      maxY = Math.max(maxY, this.mesh.nodes[i * 3 + 1]);
      maxZ = Math.max(maxZ, this.mesh.nodes[i * 3 + 2]);
    }
    const bboxSize = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
    const dist = camera.position.length();
    const fov = (camera.fov * Math.PI) / 180;
    const pixelSize = (2 * Math.tan(fov / 2) * dist) / this.canvas.height;
    return pixelSize * 15;
  }

  private unprojectToPlane(screenX: number, screenY: number, plane: THREE.Plane): THREE.Vector3 | null {
    const mouse = this.getNormalizedMouse({ clientX: screenX, clientY: screenY } as MouseEvent);
    this.raycaster.setFromCamera(new THREE.Vector2(mouse.x, mouse.y), this.renderer.camera);
    const intersection = new THREE.Vector3();
    return this.raycaster.ray.intersectPlane(plane, intersection);
  }

  private onMouseDown = (e: MouseEvent): void => {
    if (e.button !== 0) return;

    if (this.mode === 'fix') {
      const result = this.getNearestNode(e.clientX, e.clientY);
      if (result) {
        this.onAddBC('fix', result.nodeId);
        this.renderer.highlightNode(result.nodeId);
      }
    } else if (this.mode === 'force') {
      const result = this.getNearestNode(e.clientX, e.clientY);
      if (result) {
        this.dragState.active = true;
        this.dragState.nodeId = result.nodeId;
        this.dragState.startPoint.copy(result.point);
        this.dragState.currentPoint.copy(result.point);
        const cameraDir = new THREE.Vector3();
        this.renderer.camera.getWorldDirection(cameraDir);
        this.dragState.plane.setFromNormalAndCoplanarPoint(cameraDir.negate(), result.point);
        e.preventDefault();
      }
    }
  };

  private onMouseMove = (e: MouseEvent): void => {
    if (this.mode === 'fix') {
      const result = this.getNearestNode(e.clientX, e.clientY);
      if (result) {
        this.renderer.highlightNode(result.nodeId);
        this.canvas.style.cursor = 'pointer';
      } else {
        this.renderer.clearHighlight();
        this.canvas.style.cursor = 'crosshair';
      }
    } else if (this.mode === 'force' && this.dragState.active) {
      const intersection = this.unprojectToPlane(e.clientX, e.clientY, this.dragState.plane);
      if (intersection) {
        this.dragState.currentPoint.copy(intersection);
        this.renderer.showForceArrow(this.dragState.startPoint, this.dragState.currentPoint);
      }
      e.preventDefault();
    }
  };

  private onMouseUp = (e: MouseEvent): void => {
    if (!this.dragState.active) return;

    if (this.mode === 'force' && e.button === 0) {
      const dragVec = new THREE.Vector3()
        .subVectors(this.dragState.currentPoint, this.dragState.startPoint);
      const magnitude = dragVec.length() * 10;

      if (magnitude > 0.01) {
        const forceDir = dragVec.clone().normalize();
        const force: Vec3 = {
          x: forceDir.x * magnitude,
          y: forceDir.y * magnitude,
          z: forceDir.z * magnitude,
        };
        this.onAddBC('force', this.dragState.nodeId, force);
      }
    }

    this.cancelDrag();
  };

  private cancelDrag(): void {
    this.dragState.active = false;
    this.dragState.nodeId = -1;
    this.renderer.hideForceArrow();
    this.renderer.clearHighlight();
  }
}
