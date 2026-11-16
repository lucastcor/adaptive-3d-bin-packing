import { Box, Layers } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { ProblemInstance, SolverResult } from "../types";

type PackingViewerProps = {
  instance: ProblemInstance | null;
  result: SolverResult | null;
};

function colorForId(id: string) {
  let hash = 0;
  for (let index = 0; index < id.length; index += 1) {
    hash = (hash * 31 + id.charCodeAt(index)) >>> 0;
  }
  const hue = hash % 360;
  return new THREE.Color(`hsl(${hue}, 58%, 54%)`);
}

export function PackingViewer({ instance, result }: PackingViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [selectedBin, setSelectedBin] = useState(0);
  const binCount = useMemo(() => Math.max(1, result?.bin_metrics.length ?? 1), [result]);

  useEffect(() => {
    if (selectedBin >= binCount) {
      setSelectedBin(0);
    }
  }, [binCount, selectedBin]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !instance) {
      return;
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#f8fafc");

    const width = container.clientWidth || 640;
    const height = container.clientHeight || 420;
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 5000);
    const sceneLength = instance.bin.length;
    const sceneDepth = instance.bin.width;
    const sceneHeight = instance.bin.height;
    const maxDimension = Math.max(sceneLength, sceneDepth, sceneHeight);
    const center = new THREE.Vector3(sceneLength / 2, sceneHeight / 2, sceneDepth / 2);
    const sceneRadius = Math.max(
      Math.sqrt(sceneLength ** 2 + sceneDepth ** 2 + sceneHeight ** 2) / 2,
      1
    );
    const verticalFov = THREE.MathUtils.degToRad(camera.fov);
    const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * camera.aspect);
    const fitDistance = sceneRadius / Math.sin(Math.min(verticalFov, horizontalFov) / 2);
    const cameraDirection = new THREE.Vector3(1.15, 0.82, 1.35).normalize();
    camera.near = Math.max(sceneRadius / 100, 0.05);
    camera.far = Math.max(fitDistance * 8, maxDimension * 12);
    camera.position.copy(center).add(cameraDirection.multiplyScalar(fitDistance * 1.2));
    camera.updateProjectionMatrix();
    camera.lookAt(center);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    container.replaceChildren(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.copy(center);
    controls.minDistance = sceneRadius * 0.35;
    controls.maxDistance = fitDistance * 4;

    const ambient = new THREE.HemisphereLight(0xffffff, 0xd1d5db, 2.2);
    scene.add(ambient);
    const key = new THREE.DirectionalLight(0xffffff, 2.4);
    key.position.set(maxDimension, maxDimension * 1.8, maxDimension * 1.2);
    scene.add(key);

    // Domain coordinates are X=length, Y=width/depth, Z=height. Three.js is Y-up.
    const binGeometry = new THREE.BoxGeometry(sceneLength, sceneHeight, sceneDepth);
    const wireframe = new THREE.LineSegments(
      new THREE.EdgesGeometry(binGeometry),
      new THREE.LineBasicMaterial({ color: "#334155", linewidth: 1 })
    );
    wireframe.position.copy(center);
    scene.add(wireframe);

    const grid = new THREE.GridHelper(Math.max(sceneLength, sceneDepth) * 1.2, 10, "#94a3b8", "#dbe2ea");
    grid.position.set(sceneLength / 2, 0, sceneDepth / 2);
    scene.add(grid);

    const itemGeometries: THREE.BufferGeometry[] = [];
    const placements = result?.placements.filter((placement) => placement.bin_id === selectedBin) ?? [];
    for (const placement of placements) {
      const geometry = new THREE.BoxGeometry(
        placement.placed_length,
        placement.placed_height,
        placement.placed_width
      );
      itemGeometries.push(geometry);
      const material = new THREE.MeshStandardMaterial({
        color: colorForId(placement.item_id),
        roughness: 0.55,
        metalness: 0.08,
        transparent: true,
        opacity: 0.9
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(
        placement.x + placement.placed_length / 2,
        placement.z + placement.placed_height / 2,
        placement.y + placement.placed_width / 2
      );
      scene.add(mesh);
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(geometry),
        new THREE.LineBasicMaterial({ color: "#0f172a", transparent: true, opacity: 0.35 })
      );
      edges.position.copy(mesh.position);
      scene.add(edges);
    }

    const resizeObserver = new ResizeObserver(([entry]) => {
      const nextWidth = entry.contentRect.width;
      const nextHeight = entry.contentRect.height;
      camera.aspect = nextWidth / Math.max(nextHeight, 1);
      camera.updateProjectionMatrix();
      renderer.setSize(nextWidth, nextHeight);
    });
    resizeObserver.observe(container);

    let frame = 0;
    const render = () => {
      controls.update();
      renderer.render(scene, camera);
      frame = requestAnimationFrame(render);
    };
    render();

    return () => {
      cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      controls.dispose();
      renderer.dispose();
      binGeometry.dispose();
      for (const geometry of itemGeometries) {
        geometry.dispose();
      }
      container.replaceChildren();
    };
  }, [instance, result, selectedBin]);

  return (
    <section className="viewer-shell" aria-label="Visualização 3D">
      <div className="viewer-toolbar">
        <div className="panel-heading">
          <Box size={18} aria-hidden="true" />
          <h2>3D Bin</h2>
        </div>
        <div className="segmented" role="tablist" aria-label="Caixas">
          {Array.from({ length: binCount }, (_, index) => (
            <button
              key={index}
              type="button"
              className={selectedBin === index ? "active" : ""}
              onClick={() => setSelectedBin(index)}
              role="tab"
              aria-selected={selectedBin === index}
              title={`Bin ${index + 1}`}
            >
              <Layers size={15} aria-hidden="true" />
              <span>{index + 1}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="viewer-canvas" ref={containerRef} />
    </section>
  );
}
