"use client";

import React from "react";

export default function ShapWaterfall({ contributions }: { contributions: { feature: string; impact: number }[] }) {
  return (
    <div className="bg-slate-950 border border-slate-800 rounded-xl p-4">
      <h4 className="text-xs uppercase font-bold tracking-wider text-slate-400 mb-3">Feature Attribution (SHAP Vectors)</h4>
      <div className="space-y-2">
        {contributions.slice(0, 5).map((item, idx) => (
          <div key={idx} className="text-xs">
            <div className="flex justify-between text-slate-300 mb-0.5">
              <span>{item.feature}</span>
              <span className="font-mono">{item.impact.toFixed(3)}</span>
            </div>
            <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
              <div
                className={`h-full ${item.impact >= 0 ? "bg-rose-500" : "bg-emerald-500"}`}
                style={{ width: `${Math.min(Math.abs(item.impact) * 100, 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
