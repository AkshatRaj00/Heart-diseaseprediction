"use client";

import React, { useState } from "react";

export interface PatientData {
  age: number;
  sex: number;
  cp: number;
  trestbps: number;
  chol: number;
  fbs: number;
  restecg: number;
  thalach: number;
  exang: number;
  oldpeak: number;
  slope: number;
  ca: number;
  thal: number;
}

export default function TelemetryForm({ onSubmit, loading }: { onSubmit: (data: PatientData) => void, loading: boolean }) {
  const [form, setForm] = useState<PatientData>({
    age: 54, sex: 1, cp: 2, trestbps: 130, chol: 240, fbs: 0,
    restecg: 1, thalach: 150, exang: 0, oldpeak: 1.0, slope: 1, ca: 0, thal: 2
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: parseFloat(value) }));
  };

  return (
    <form onSubmit={(e) => { e.preventDefault(); onSubmit(form); }} className="grid grid-cols-2 gap-3 text-xs">
      <div>
        <label className="text-slate-400">Age</label>
        <input type="number" name="age" value={form.age} onChange={handleChange} className="w-full bg-slate-900 border border-slate-800 rounded p-1.5 text-white" />
      </div>
      <div>
        <label className="text-slate-400">Sex (0=F, 1=M)</label>
        <select name="sex" value={form.sex} onChange={handleChange} className="w-full bg-slate-900 border border-slate-800 rounded p-1.5 text-white">
          <option value="1">Male</option>
          <option value="0">Female</option>
        </select>
      </div>
      <div>
        <label className="text-slate-400">Resting BP (mm Hg)</label>
        <input type="number" name="trestbps" value={form.trestbps} onChange={handleChange} className="w-full bg-slate-900 border border-slate-800 rounded p-1.5 text-white" />
      </div>
      <div>
        <label className="text-slate-400">Cholesterol (mg/dl)</label>
        <input type="number" name="chol" value={form.chol} onChange={handleChange} className="w-full bg-slate-900 border border-slate-800 rounded p-1.5 text-white" />
      </div>
      <div className="col-span-2">
        <button type="submit" disabled={loading} className="w-full py-2.5 rounded-lg bg-rose-600 hover:bg-rose-700 text-white font-medium transition">
          {loading ? "Running ONNX Engine..." : "Evaluate Cardiac Vector"}
        </button>
      </div>
    </form>
  );
}
