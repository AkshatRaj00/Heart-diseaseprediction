"use client";

import React, { useEffect, useRef } from "react";
import physioData from "./real_physionet_ecg.json";

interface EcgProps {
  hr: number;
  stDepression: number;
  isIschemic: boolean;
  bp?: number;
}

export default function EcgMonitor({ hr, stDepression, isIschemic, bp = 120 }: EcgProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const isAsystole = hr <= 10 || bp <= 20;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let x = 0;
    const width = canvas.width;
    const height = canvas.height;
    const baselineY = height * 0.62;

    const sampleArray: number[] = isIschemic ? physioData.ischemic_mv : physioData.normal_mv;
    const totalSamples = sampleArray.length;

    // Draw Static Grid
    ctx.fillStyle = isAsystole ? "#150505" : "#03070d";
    ctx.fillRect(0, 0, width, height);

    for (let i = 0; i < width; i += 16) {
      ctx.strokeStyle = isAsystole ? "rgba(248, 81, 73, 0.12)" : "rgba(46, 160, 67, 0.06)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, height);
      ctx.stroke();
    }
    for (let j = 0; j < height; j += 16) {
      ctx.strokeStyle = isAsystole ? "rgba(248, 81, 73, 0.12)" : "rgba(46, 160, 67, 0.06)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, j);
      ctx.lineTo(width, j);
      ctx.stroke();
    }

    let sampleIdx = 0;
    const speed = isAsystole ? 2.0 : Math.max(1.5, (hr / 60) * 1.8);
    let prevY = baselineY;

    const render = () => {
      ctx.fillStyle = isAsystole ? "#150505" : "#03070d";
      ctx.fillRect(x, 0, 14, height);

      ctx.strokeStyle = isAsystole ? "rgba(248, 81, 73, 0.15)" : "rgba(46, 160, 67, 0.08)";
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();

      let currentY = baselineY;

      if (isAsystole) {
        // Clinical Asystole: Isoelectric flatline with minor electrical baseline noise (<0.02mV)
        const electricalNoise = (Math.random() - 0.5) * 1.8;
        currentY = baselineY + electricalNoise;
      } else {
        const rawVoltage = sampleArray[Math.floor(sampleIdx) % totalSamples];
        currentY = baselineY - (rawVoltage * 55);
        sampleIdx = (sampleIdx + 0.65) % totalSamples;
      }

      ctx.strokeStyle = isAsystole ? "#ff3333" : (isIschemic ? "#f85149" : "#3fb950");
      ctx.lineWidth = isAsystole ? 2.5 : 2;
      ctx.lineJoin = "round";

      ctx.beginPath();
      ctx.moveTo(x === 0 ? 0 : x - speed, prevY);
      ctx.lineTo(x, currentY);
      ctx.stroke();

      prevY = currentY;
      x += speed;

      if (x >= width) {
        x = 0;
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [hr, stDepression, isIschemic, isAsystole, bp]);

  return (
    <div className={`relative border rounded-lg overflow-hidden shadow-inner transition-colors duration-500 ${
      isAsystole ? "border-[#f85149] bg-[#150505] shadow-[#f85149]/20 shadow-lg" : "border-[#30363d] bg-[#03070d]"
    }`}>
      <div className="absolute top-2 left-3 z-10 flex flex-wrap items-center gap-3 text-[10px] font-mono text-[#8b949e]">
        <span className="flex items-center gap-1.5 font-bold text-white">
          <span className={`h-2.5 w-2.5 rounded-full ${isAsystole ? "bg-[#f85149] animate-ping" : (isIschemic ? "bg-[#f85149]" : "bg-[#3fb950]")}`} />
          {isAsystole ? "CRITICAL ALERT: ASYSTOLE / CARDIAC ARREST" : "PHYSIO-NET LEAD-II &bull; 250Hz SAMPLES"}
        </span>
        <span>PATIENT HR: <strong className={isAsystole ? "text-[#f85149] font-bold" : "text-white"}>{hr} BPM</strong></span>
        <span>BP: <strong className={isAsystole ? "text-[#f85149] font-bold" : "text-white"}>{bp} mmHg</strong></span>
        <span>
          STATUS: <strong className={isAsystole ? "text-[#f85149] animate-pulse" : (isIschemic ? "text-[#f85149]" : "text-[#3fb950]")}>
            {isAsystole ? "ISOELECTRIC FLATLINE (PULSELESS)" : (isIschemic ? "Record s20011 (Ischemia)" : "Record 16265 (Sinus)")}
          </strong>
        </span>
      </div>
      <canvas ref={canvasRef} width={760} height={150} className="w-full h-[140px] block" />
    </div>
  );
}
