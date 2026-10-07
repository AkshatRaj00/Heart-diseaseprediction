"use client";

import React, { useEffect, useRef } from "react";
import * as THREE from "three";

export default function HeartCanvas({ riskScore }: { riskScore: number }) {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 320;
    const height = container.clientHeight || 280;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 50);
    camera.position.set(0, 0, 4.0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.replaceChildren(renderer.domElement);

    const heartGroup = new THREE.Group();

    // Clinical Alert Coloration
    const heartColor = riskScore >= 70 ? 0xf85149 : riskScore >= 35 ? 0xd29922 : 0x3fb950;

    const myocardialMat = new THREE.MeshStandardMaterial({
      color: heartColor,
      roughness: 0.35,
      metalness: 0.3,
    });

    // Ventricular Apex Base
    const ventGeom = new THREE.SphereGeometry(0.85, 32, 32);
    ventGeom.scale(0.8, 1.35, 0.75);
    const ventricle = new THREE.Mesh(ventGeom, myocardialMat);
    ventricle.position.set(0, -0.25, 0);
    ventricle.rotation.z = 0.15;
    heartGroup.add(ventricle);

    // Left Atrium Chamber
    const leftAtriumGeom = new THREE.SphereGeometry(0.52, 24, 24);
    leftAtriumGeom.scale(1.05, 0.9, 0.85);
    const leftAtrium = new THREE.Mesh(leftAtriumGeom, myocardialMat);
    leftAtrium.position.set(-0.42, 0.55, -0.1);
    heartGroup.add(leftAtrium);

    // Right Atrium Chamber
    const rightAtriumGeom = new THREE.SphereGeometry(0.48, 24, 24);
    const rightAtrium = new THREE.Mesh(rightAtriumGeom, myocardialMat);
    rightAtrium.position.set(0.42, 0.5, -0.1);
    heartGroup.add(rightAtrium);

    // Aortic Trunk
    const aortaGeom = new THREE.CylinderGeometry(0.16, 0.20, 0.65, 16);
    const aortaMat = new THREE.MeshStandardMaterial({ color: 0x8b949e, roughness: 0.4, metalness: 0.5 });
    const aorta = new THREE.Mesh(aortaGeom, aortaMat);
    aorta.position.set(0.04, 0.95, -0.05);
    aorta.rotation.z = -0.25;
    heartGroup.add(aorta);

    scene.add(heartGroup);

    // Dynamic Lighting
    const amb = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(amb);

    const dir1 = new THREE.DirectionalLight(0xffffff, 1.8);
    dir1.position.set(4, 5, 4);
    scene.add(dir1);

    const dir2 = new THREE.DirectionalLight(0x58a6ff, 0.8);
    dir2.position.set(-4, -3, -2);
    scene.add(dir2);

    let animId: number;
    const clock = new THREE.Clock();
    const bpm = Math.round(60 + (riskScore / 100) * 60);
    const pulseFreq = (bpm / 60) * Math.PI * 2;

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const t = clock.getElapsedTime();

      // Realistic non-linear cardiac pulse
      const systolic = Math.pow(Math.max(0, Math.sin(t * pulseFreq)), 4) * 0.15;
      const s = 1.0 + systolic;
      heartGroup.scale.set(s, s * 0.97, s);

      heartGroup.rotation.y = Math.sin(t * 0.8) * 0.25;
      heartGroup.rotation.x = 0.1 + Math.cos(t * 0.5) * 0.08;

      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", handleResize);
      renderer.dispose();
      container.replaceChildren();
    };
  }, [riskScore]);

  return (
    <div className="h-64 w-full rounded-md bg-[#0d1117] border border-[#30363d] flex items-center justify-center overflow-hidden relative shadow-inner">
      <div className="absolute top-2.5 left-3 text-[10px] font-mono text-[#8b949e] uppercase tracking-wider z-10 flex items-center gap-1.5">
        <span className="h-2 w-2 rounded-full bg-[#f85149] animate-ping" />
        Hemodynamics: {Math.round(60 + (riskScore / 100) * 60)} BPM &bull; Real Myocardium
      </div>
      <div ref={mountRef} className="w-full h-full" />
    </div>
  );
}
