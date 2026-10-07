"use client";

import React, { useState, useRef, useEffect } from "react";
import HeartCanvas from "../components/HeartCanvas";
import EcgMonitor from "../components/EcgMonitor";
import { 
  Activity, ShieldAlert, Cpu, Stethoscope, 
  Send, ChevronRight, CheckCircle2, Sparkles, Sliders, RefreshCw, 
  Printer, Zap, Users, HeartPulse, FileText, Upload, AlertTriangle, FileSpreadsheet,
  Pill, Apple, Skull, AlertOctagon, Mic, MicOff, Volume2, Database, Gauge, Clock
} from "lucide-react";

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

export default function ClinicalEnterpriseDashboard() {
  const [activeTab, setActiveTab] = useState<"matrix" | "ecg" | "xai" | "board" | "rx" | "prognosis" | "pump" | "cohort">("matrix");
  const [form, setForm] = useState<PatientData>({
    age: 67, sex: 1, cp: 3, trestbps: 160, chol: 286, fbs: 0,
    restecg: 2, thalach: 108, exang: 1, oldpeak: 2.6, slope: 1, ca: 3, thal: 2
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [boardStreamText, setBoardStreamText] = useState("");
  const [isBoardStreaming, setIsBoardStreaming] = useState(false);

  // Prognosis and Pump States
  const [prognosis, setPrognosis] = useState<any>(null);
  const [pumpData, setPumpData] = useState<any>(null);
  const [targetBp, setTargetBp] = useState<number>(120);

  // Voice AI Agent States
  const [isListening, setIsListening] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(true);

  // Asystole detection
  const isAsystole = form.thalach <= 10 || form.trestbps <= 20 || form.age <= 0;

  // Batch CSV Cohort State
  const [cohort, setCohort] = useState<any[]>([]);
  const [cohortLoading, setCohortLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [chatMessages, setChatMessages] = useState<Array<{ role: "assistant" | "user"; content: string }>>([
    {
      role: "assistant",
      content: "CardioSense ICU Autonomous Core online. 60-Minute Prognosis, Closed-Loop IV Pump, Voice AI and FHIR Export fully active."
    }
  ]);
  const [inputMessage, setInputMessage] = useState("");
  const [chatStreaming, setChatStreaming] = useState(false);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages, boardStreamText]);

  // Text to Speech
  const speakVoice = (text: string) => {
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const cleanText = text.replace(/###/g, "").replace(/\*\*/g, "").substring(0, 200);
      const utterance = new SpeechSynthesisUtterance(cleanText);
      utterance.rate = 1.05;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  };

  // Voice Recognition (Hands-Free Dictation)
  const toggleVoice = () => {
    if (typeof window === "undefined") return;
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Web Speech API not supported in this browser. Please use Chrome/Edge.");
      setSpeechSupported(false);
      return;
    }

    if (isListening) {
      setIsListening(false);
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      recognition.lang = "en-US";
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;

      recognition.onstart = () => setIsListening(true);
      recognition.onend = () => setIsListening(false);
      recognition.onerror = () => setIsListening(false);

      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setInputMessage(transcript);
        speakVoice(`Voice command acknowledged: ${transcript}`);
      };

      recognition.start();
    } catch {
      setIsListening(false);
    }
  };

  const loadPreset = (type: "normal" | "high_risk" | "asystole") => {
    if (type === "normal") {
      setForm({
        age: 44, sex: 0, cp: 2, trestbps: 108, chol: 141, fbs: 0,
        restecg: 0, thalach: 175, exang: 0, oldpeak: 0.0, slope: 0, ca: 0, thal: 1
      });
    } else if (type === "high_risk") {
      setForm({
        age: 67, sex: 1, cp: 3, trestbps: 160, chol: 286, fbs: 0,
        restecg: 2, thalach: 108, exang: 1, oldpeak: 2.6, slope: 1, ca: 3, thal: 2
      });
    } else {
      setForm({
        age: 72, sex: 1, cp: 3, trestbps: 0, chol: 0, fbs: 0,
        restecg: 0, thalach: 0, exang: 0, oldpeak: 0.0, slope: 1, ca: 0, thal: 1
      });
    }
  };

  const handlePredict = async () => {
    setLoading(true);
    setBoardStreamText("");
    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form)
      });
      if (!res.ok) throw new Error("Inference failed");
      const data = await res.json();
      setResult(data);
      setLoading(false);

      // Fetch 60-Minute Prognosis Forecast
      fetch("http://127.0.0.1:8000/prognosis/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ telemetry: form, risk_score: data.risk_score_percentage })
      }).then(r => r.json()).then(d => setPrognosis(d)).catch(() => {});

      // Fetch Closed-Loop IV Titration
      fetch("http://127.0.0.1:8000/pump/titrate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ telemetry: form, target_bp: targetBp, weight_kg: 70 })
      }).then(r => r.json()).then(d => setPumpData(d)).catch(() => {});

      // Trigger Consensus Stream
      setIsBoardStreaming(true);
      const streamRes = await fetch("http://127.0.0.1:8000/agent/stream-board", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          telemetry: form,
          risk_score: data.risk_score_percentage,
          risk_tier: data.risk_tier,
          shaps: data.shap_contributions,
          recourse: data.counterfactual_recourse
        })
      });

      if (!streamRes.body) return;
      const reader = streamRes.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        const lines = chunk.split("\n\n");
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const raw = line.replace("data: ", "").trim();
            if (raw === "[DONE]") break;
            try {
              const parsed = JSON.parse(raw);
              if (parsed.token) {
                setBoardStreamText(prev => prev + parsed.token);
              }
            } catch {}
          }
        }
      }
      setIsBoardStreaming(false);
    } catch {
      alert("Inference Server unreachable on port 8000.");
      setLoading(false);
      setIsBoardStreaming(false);
    }
  };

  // FHIR Export Handler
  const handleFhirExport = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/fhir/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ telemetry: form, risk_score: result?.risk_score_percentage ?? 83.2 })
      });
      const bundle = await res.json();
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(bundle, null, 2));
      const downloadAnchor = document.createElement("a");
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", `FHIR_R4_Bundle_${new Date().toISOString().slice(0,10)}.json`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
    } catch {
      alert("Failed to export FHIR bundle.");
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setCohortLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/batch-triage", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      setCohort(data.patients || []);
      setActiveTab("cohort");
    } catch {
      alert("Failed to parse and triage CSV file. Ensure valid columns.");
    } finally {
      setCohortLoading(false);
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim() || chatStreaming) return;

    const userText = inputMessage;
    setInputMessage("");
    setChatMessages(prev => [
      ...prev, 
      { role: "user", content: userText },
      { role: "assistant", content: "" }
    ]);
    setChatStreaming(true);

    try {
      const res = await fetch("http://127.0.0.1:8000/agent/stream-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userText,
          context: {
            telemetry: form,
            risk_score: result?.risk_score_percentage ?? 83.2,
            top_driver: result?.shap_contributions?.[0]?.feature ?? "ischemia"
          }
        })
      });

      if (!res.body) return;
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let fullReply = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        const lines = chunk.split("\n\n");
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const raw = line.replace("data: ", "").trim();
            if (raw === "[DONE]") break;
            try {
              const parsed = JSON.parse(raw);
              if (parsed.token) {
                fullReply += parsed.token;
                setChatMessages(prev => {
                  const updated = [...prev];
                  updated[updated.length - 1].content += parsed.token;
                  return updated;
                });
              }
            } catch {}
          }
        }
      }
      speakVoice(fullReply);
    } catch {
      setChatMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1].content = "Clinical copilot stream interrupted.";
        return updated;
      });
    } finally {
      setChatStreaming(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#090d13] text-[#c9d1d9] font-sans antialiased selection:bg-[#58a6ff]/20">
      
      {/* Code Blue Alert Bar */}
      {isAsystole && (
        <div className="bg-[#b62324] text-white px-4 py-2 font-mono text-xs font-bold flex items-center justify-between animate-pulse">
          <div className="flex items-center gap-2">
            <AlertOctagon className="w-5 h-5" />
            <span>CRITICAL CODE BLUE: PATIENT HEMODYNAMIC COLLAPSE (ASYSTOLE). BEGIN ACLS COMPRESSIONS.</span>
          </div>
          <span className="bg-black/40 px-2 py-0.5 rounded text-[10px]">DEFIBRILLATION WITHHELD</span>
        </div>
      )}

      {/* Top Navigation */}
      <header className="border-b border-[#21262d] bg-[#161b22]/90 backdrop-blur-md sticky top-0 z-50 px-4 lg:px-8 py-2.5">
        <div className="max-w-[1600px] mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className={`h-9 w-9 rounded-lg border flex items-center justify-center shadow-inner ${
              isAsystole ? "bg-[#b62324]/20 border-[#f85149] text-[#f85149]" : "bg-[#21262d] border-[#30363d] text-[#f85149]"
            }`}>
              {isAsystole ? <Skull className="w-5 h-5 animate-bounce" /> : <HeartPulse className="w-5 h-5 animate-pulse" />}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm font-bold text-white tracking-tight">CardioSense Autonomous Core</h1>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[#1f6feb]/15 text-[#58a6ff] border border-[#1f6feb]/30 font-medium">
                  FHIR R4 Ready
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[#3fb950]/15 text-[#3fb950] border border-[#3fb950]/30 font-medium hidden sm:inline-block">
                  Voice AI Active
                </span>
              </div>
              <p className="text-[11px] font-mono text-[#8b949e]">Closed-Loop Autotitration &bull; Predictive 60m Prognosis</p>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <button
              onClick={handleFhirExport}
              className="text-[11px] px-3 py-1.5 rounded-md bg-[#21262d] hover:bg-[#30363d] border border-[#30363d] text-[#58a6ff] transition font-medium flex items-center gap-1.5"
            >
              <Database className="w-3.5 h-3.5" />
              <span>Export FHIR R4</span>
            </button>

            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileUpload} 
              accept=".csv" 
              className="hidden" 
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={cohortLoading}
              className="text-[11px] px-3 py-1.5 rounded-md bg-[#21262d] hover:bg-[#30363d] border border-[#30363d] text-white transition font-medium flex items-center gap-1.5"
            >
              <Upload className="w-3.5 h-3.5 text-[#58a6ff]" />
              <span>{cohortLoading ? "Processing..." : "Upload CSV"}</span>
            </button>

            <button
              onClick={() => loadPreset("normal")}
              className="text-[11px] px-2.5 py-1.5 rounded-md bg-[#21262d] hover:bg-[#30363d] border border-[#30363d] text-[#c9d1d9] transition font-medium"
            >
              Normal
            </button>
            <button
              onClick={() => loadPreset("high_risk")}
              className="text-[11px] px-2.5 py-1.5 rounded-md bg-[#f85149]/10 hover:bg-[#f85149]/20 border border-[#f85149]/30 text-[#f85149] transition font-medium"
            >
              Ischemia
            </button>
            <button
              onClick={() => loadPreset("asystole")}
              className="text-[11px] px-2.5 py-1.5 rounded-md bg-[#b62324]/20 hover:bg-[#b62324]/40 border border-[#f85149]/50 text-[#f85149] transition font-medium flex items-center gap-1"
            >
              <Skull className="w-3 h-3" />
              <span>Arrest</span>
            </button>
            <button
              onClick={() => window.print()}
              className="text-[11px] px-2.5 py-1.5 rounded-md bg-[#1f6feb]/10 hover:bg-[#1f6feb]/20 border border-[#1f6feb]/30 text-[#58a6ff] transition font-medium flex items-center gap-1"
            >
              <Printer className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <div className="max-w-[1600px] mx-auto px-4 lg:px-8 py-5 flex flex-col gap-5">
        
        {/* Lead-II ECG */}
        <section>
          <EcgMonitor 
            hr={form.thalach} 
            stDepression={form.oldpeak} 
            isIschemic={(result ? result.has_heart_disease : true)} 
            bp={form.trestbps}
          />
        </section>

        {/* Status Metrics */}
        <section className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
          <div className="p-3 rounded-lg bg-[#161b22] border border-[#30363d]">
            <div className="text-[10px] font-mono uppercase text-[#8b949e]">Ischemic Risk</div>
            <div className={`text-2xl font-black font-mono mt-0.5 ${isAsystole ? "text-[#f85149]" : "text-white"}`}>
              {isAsystole ? "100.0%" : (result ? `${result.risk_score_percentage}%` : "83.20%")}
            </div>
          </div>

          <div className="p-3 rounded-lg bg-[#161b22] border border-[#30363d]">
            <div className="text-[10px] font-mono uppercase text-[#8b949e]">Telemetry Triage</div>
            <div className={`text-xs font-bold font-mono mt-1.5 flex items-center gap-1 ${
              isAsystole ? "text-[#f85149]" : ((result ? result.has_heart_disease : true) ? "text-[#f85149]" : "text-[#3fb950]")
            }`}>
              <span className={`h-2 w-2 rounded-full ${
                isAsystole ? "bg-[#f85149] animate-ping" : ((result ? result.has_heart_disease : true) ? "bg-[#f85149]" : "bg-[#3fb950]")
              }`} />
              {isAsystole ? "CODE BLUE" : ((result ? result.has_heart_disease : true) ? "CRITICAL RED" : "NORMAL")}
            </div>
          </div>

          <div className="p-3 rounded-lg bg-[#161b22] border border-[#30363d]">
            <div className="text-[10px] font-mono uppercase text-[#8b949e]">Workload (RPP)</div>
            <div className="text-sm font-bold font-mono text-[#58a6ff] mt-1">
              {form.trestbps * form.thalach} mmHg&bull;bpm
            </div>
          </div>

          <div className="p-3 rounded-lg bg-[#161b22] border border-[#30363d]">
            <div className="text-[10px] font-mono uppercase text-[#8b949e]">Pump Delivery</div>
            <div className="text-sm font-bold font-mono text-[#3fb950] mt-1">
              {pumpData ? `${pumpData.infusion_rate_ml_hr} mL/hr` : "18.0 mL/hr"}
            </div>
          </div>

          <div className="p-3 rounded-lg bg-[#161b22] border border-[#30363d] hidden lg:block">
            <div className="text-[10px] font-mono uppercase text-[#8b949e]">Early Prognosis</div>
            <div className="text-xs font-bold font-mono text-[#d29922] mt-1 truncate">
              {prognosis ? prognosis.rpp_workload : "Hyperdynamic Strain"}
            </div>
          </div>

          <div className="p-3 rounded-lg bg-[#161b22] border border-[#30363d] hidden lg:block">
            <div className="text-[10px] font-mono uppercase text-[#8b949e]">Active Unit</div>
            <div className="text-xs font-mono text-[#58a6ff] mt-1">
              Smart ICU Bay #01
            </div>
          </div>
        </section>

        {/* Tab Navigation */}
        <div className="flex border-b border-[#21262d] gap-2 overflow-x-auto text-xs font-mono">
          <button
            onClick={() => setActiveTab("matrix")}
            className={`pb-2.5 px-3 flex items-center gap-2 border-b-2 font-semibold transition ${
              activeTab === "matrix" ? "border-[#58a6ff] text-white" : "border-transparent text-[#8b949e] hover:text-[#c9d1d9]"
            }`}
          >
            <Sliders className="w-4 h-4" />
            <span>1. Telemetry Matrix</span>
          </button>

          <button
            onClick={() => setActiveTab("prognosis")}
            className={`pb-2.5 px-3 flex items-center gap-2 border-b-2 font-semibold transition ${
              activeTab === "prognosis" ? "border-[#58a6ff] text-white" : "border-transparent text-[#8b949e] hover:text-[#c9d1d9]"
            }`}
          >
            <Clock className="w-4 h-4 text-[#d29922]" />
            <span>2. 60-Min Prognosis</span>
          </button>

          <button
            onClick={() => setActiveTab("pump")}
            className={`pb-2.5 px-3 flex items-center gap-2 border-b-2 font-semibold transition ${
              activeTab === "pump" ? "border-[#58a6ff] text-white" : "border-transparent text-[#8b949e] hover:text-[#c9d1d9]"
            }`}
          >
            <Gauge className="w-4 h-4 text-[#3fb950]" />
            <span>3. Autonomous IV Pump</span>
          </button>

          <button
            onClick={() => setActiveTab("ecg")}
            className={`pb-2.5 px-3 flex items-center gap-2 border-b-2 font-semibold transition ${
              activeTab === "ecg" ? "border-[#58a6ff] text-white" : "border-transparent text-[#8b949e] hover:text-[#c9d1d9]"
            }`}
          >
            <HeartPulse className="w-4 h-4" />
            <span>4. 3D Myocardium</span>
          </button>

          <button
            onClick={() => setActiveTab("xai")}
            className={`pb-2.5 px-3 flex items-center gap-2 border-b-2 font-semibold transition ${
              activeTab === "xai" ? "border-[#58a6ff] text-white" : "border-transparent text-[#8b949e] hover:text-[#c9d1d9]"
            }`}
          >
            <Cpu className="w-4 h-4" />
            <span>5. SHAP XAI</span>
          </button>

          <button
            onClick={() => setActiveTab("board")}
            className={`pb-2.5 px-3 flex items-center gap-2 border-b-2 font-semibold transition ${
              activeTab === "board" ? "border-[#58a6ff] text-white" : "border-transparent text-[#8b949e] hover:text-[#c9d1d9]"
            }`}
          >
            <Users className="w-4 h-4" />
            <span>6. Medical Board</span>
          </button>

          <button
            onClick={() => setActiveTab("rx")}
            className={`pb-2.5 px-3 flex items-center gap-2 border-b-2 font-semibold transition ${
              activeTab === "rx" ? "border-[#58a6ff] text-white" : "border-transparent text-[#8b949e] hover:text-[#c9d1d9]"
            }`}
          >
            <Pill className="w-4 h-4 text-[#3fb950]" />
            <span>7. Clinical Rx &amp; Diet</span>
          </button>

          <button
            onClick={() => setActiveTab("cohort")}
            className={`pb-2.5 px-3 flex items-center gap-2 border-b-2 font-semibold transition ${
              activeTab === "cohort" ? "border-[#58a6ff] text-white" : "border-transparent text-[#8b949e] hover:text-[#c9d1d9]"
            }`}
          >
            <FileSpreadsheet className="w-4 h-4" />
            <span>8. Cohort ({cohort.length})</span>
          </button>
        </div>

        {/* WORKSPACE 1: MATRIX */}
        {activeTab === "matrix" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
            <div className="lg:col-span-8 bg-[#161b22] border border-[#30363d] rounded-lg p-5">
              <div className="flex items-center justify-between pb-3 border-b border-[#21262d] mb-4">
                <span className="text-xs font-mono uppercase tracking-wider text-white font-bold flex items-center gap-2">
                  <FileText className="w-4 h-4 text-[#58a6ff]" />
                  Verified Clinical Telemetry Intake
                </span>
                <span className="text-[11px] font-mono text-[#8b949e]">Ground-Truth Cleveland Form</span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                <div className="p-3.5 rounded-md bg-[#0d1117] border border-[#21262d] space-y-3">
                  <div className="text-[11px] font-mono text-[#58a6ff] font-bold uppercase tracking-wider">A. Demographics &amp; Glycemic</div>
                  <div>
                    <label className="block text-[#8b949e] mb-1 font-mono text-[11px]">Age (years)</label>
                    <input
                      type="number"
                      value={form.age}
                      onChange={e => setForm({ ...form, age: parseFloat(e.target.value) || 0 })}
                      className="w-full bg-[#161b22] border border-[#30363d] rounded px-3 py-1.5 text-white font-mono focus:border-[#58a6ff] focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[#8b949e] mb-1 font-mono text-[11px]">Sex</label>
                    <select
                      value={form.sex}
                      onChange={e => setForm({ ...form, sex: parseInt(e.target.value) })}
                      className="w-full bg-[#161b22] border border-[#30363d] rounded px-3 py-1.5 text-white font-mono focus:border-[#58a6ff] focus:outline-none"
                    >
                      <option value={1}>1: Male</option>
                      <option value={0}>0: Female</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-[#8b949e] mb-1 font-mono text-[11px]">Fasting Blood Sugar (&gt;120 mg/dl)</label>
                    <select
                      value={form.fbs}
                      onChange={e => setForm({ ...form, fbs: parseInt(e.target.value) })}
                      className="w-full bg-[#161b22] border border-[#30363d] rounded px-3 py-1.5 text-white font-mono focus:border-[#58a6ff] focus:outline-none"
                    >
                      <option value={0}>0: False</option>
                      <option value={1}>1: True</option>
                    </select>
                  </div>
                </div>

                <div className="p-3.5 rounded-md bg-[#0d1117] border border-[#21262d] space-y-3">
                  <div className="text-[11px] font-mono text-[#58a6ff] font-bold uppercase tracking-wider">B. Hemodynamics &amp; Lipid</div>
                  <div>
                    <label className="block text-[#8b949e] mb-1 font-mono text-[11px]">Resting BP (mm Hg)</label>
                    <input
                      type="number"
                      value={form.trestbps}
                      onChange={e => setForm({ ...form, trestbps: parseFloat(e.target.value) || 0 })}
                      className="w-full bg-[#161b22] border border-[#30363d] rounded px-3 py-1.5 text-white font-mono focus:border-[#58a6ff] focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[#8b949e] mb-1 font-mono text-[11px]">Serum Cholesterol (mg/dl)</label>
                    <input
                      type="number"
                      value={form.chol}
                      onChange={e => setForm({ ...form, chol: parseFloat(e.target.value) || 0 })}
                      className="w-full bg-[#161b22] border border-[#30363d] rounded px-3 py-1.5 text-white font-mono focus:border-[#58a6ff] focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[#8b949e] mb-1 font-mono text-[11px]">Max HR (thalach bpm)</label>
                    <input
                      type="number"
                      value={form.thalach}
                      onChange={e => setForm({ ...form, thalach: parseFloat(e.target.value) || 0 })}
                      className="w-full bg-[#161b22] border border-[#30363d] rounded px-3 py-1.5 text-white font-mono focus:border-[#58a6ff] focus:outline-none"
                    />
                  </div>
                </div>

                <div className="p-3.5 rounded-md bg-[#0d1117] border border-[#21262d] space-y-3">
                  <div className="text-[11px] font-mono text-[#58a6ff] font-bold uppercase tracking-wider">C. ECG &amp; Angina</div>
                  <div>
                    <label className="block text-[#8b949e] mb-1 font-mono text-[11px]">Chest Pain Presentation (cp)</label>
                    <select
                      value={form.cp}
                      onChange={e => setForm({ ...form, cp: parseInt(e.target.value) })}
                      className="w-full bg-[#161b22] border border-[#30363d] rounded px-3 py-1.5 text-white font-mono focus:border-[#58a6ff] focus:outline-none"
                    >
                      <option value={0}>0: Typical Angina</option>
                      <option value={1}>1: Atypical Angina</option>
                      <option value={2}>2: Non-anginal Pain</option>
                      <option value={3}>3: Asymptomatic (Silent)</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-[#8b949e] mb-1 font-mono text-[11px]">Exercise Angina (exang)</label>
                    <select
                      value={form.exang}
                      onChange={e => setForm({ ...form, exang: parseInt(e.target.value) })}
                      className="w-full bg-[#161b22] border border-[#30363d] rounded px-3 py-1.5 text-white font-mono focus:border-[#58a6ff] focus:outline-none"
                    >
                      <option value={0}>0: No</option>
                      <option value={1}>1: Yes</option>
                    </select>
                  </div>
                </div>

                <div className="p-3.5 rounded-md bg-[#0d1117] border border-[#21262d] space-y-3">
                  <div className="text-[11px] font-mono text-[#58a6ff] font-bold uppercase tracking-wider">D. Perfusion &amp; Fluoroscopy</div>
                  <div>
                    <label className="block text-[#8b949e] mb-1 font-mono text-[11px]">ST Depression (oldpeak mm)</label>
                    <input
                      type="number"
                      step="0.1"
                      value={form.oldpeak}
                      onChange={e => setForm({ ...form, oldpeak: parseFloat(e.target.value) || 0 })}
                      className="w-full bg-[#161b22] border border-[#30363d] rounded px-3 py-1.5 text-white font-mono focus:border-[#58a6ff] focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[#8b949e] mb-1 font-mono text-[11px]">Fluoroscopy Vessels (ca)</label>
                    <input
                      type="number"
                      min="0" max="3"
                      value={form.ca}
                      onChange={e => setForm({ ...form, ca: parseInt(e.target.value) || 0 })}
                      className="w-full bg-[#161b22] border border-[#30363d] rounded px-3 py-1.5 text-white font-mono focus:border-[#58a6ff] focus:outline-none"
                    />
                  </div>
                </div>
              </div>

              <div className="mt-5 pt-3 border-t border-[#21262d] flex items-center justify-between">
                <span className="text-[11px] font-mono text-[#8b949e]">
                  Triggers Dual-Engine Inferences &bull; 60m Prognosis &bull; Closed-Loop Titration
                </span>
                <button
                  type="button"
                  onClick={handlePredict}
                  disabled={loading || isBoardStreaming}
                  className={`px-5 py-2.5 rounded-md font-semibold text-xs transition flex items-center gap-2 shadow-sm font-mono ${
                    isAsystole ? "bg-[#b62324] hover:bg-[#d02829] text-white" : "bg-[#238636] hover:bg-[#2ea043] text-white"
                  }`}
                >
                  {loading || isBoardStreaming ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span>{isAsystole ? "Executing Resuscitation..." : "Computing AI Pipeline..."}</span>
                    </>
                  ) : (
                    <>
                      <span>{isAsystole ? "Execute Emergency Resuscitation" : "Execute Full Diagnostic Pipeline"}</span>
                      <ChevronRight className="w-4 h-4" />
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Snapshot */}
            <div className="lg:col-span-4 flex flex-col gap-4">
              <div className="bg-[#161b22] border border-[#30363d] rounded-lg p-5">
                <span className="text-xs font-mono uppercase tracking-wider text-white font-bold block mb-3">
                  Telemetry Snapshot
                </span>
                <div className="space-y-2 text-xs font-mono">
                  <div className="flex justify-between py-1 border-b border-[#21262d]">
                    <span className="text-[#8b949e]">Hemodynamic Strain:</span>
                    <span className={form.trestbps <= 20 ? "text-[#f85149] font-bold" : "text-white"}>{form.trestbps} mmHg / {form.thalach} bpm</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#21262d]">
                    <span className="text-[#8b949e]">Lipid Profile:</span>
                    <span className="text-white font-semibold">{form.chol} mg/dL</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#21262d]">
                    <span className="text-[#8b949e]">Ischemic Deficit:</span>
                    <span className="text-[#f85149] font-semibold">{form.oldpeak} mm ST Dep</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-[#8b949e]">Occluded Arteries:</span>
                    <span className="text-[#d29922] font-semibold">{form.ca} Vessels Detected</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* WORKSPACE 2: 60-MINUTE PROGNOSIS FORECAST */}
        {activeTab === "prognosis" && (
          <div className="bg-[#161b22] border border-[#30363d] rounded-lg p-5">
            <div className="flex items-center justify-between pb-3 border-b border-[#21262d] mb-4">
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-[#d29922]" />
                <span className="text-xs font-mono uppercase tracking-wider text-white font-bold">
                  60-Minute Myocardial Infarction Early Warning Engine
                </span>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#d29922]/15 text-[#d29922] border border-[#d29922]/30 font-bold">
                Rate-Pressure Autoregressive Model
              </span>
            </div>

            <div className="p-3 mb-4 rounded-md bg-[#0d1117] border border-[#21262d] flex items-center justify-between text-xs font-mono">
              <div>
                <span className="text-[#8b949e]">Rate-Pressure Product (RPP): </span>
                <span className="text-[#58a6ff] font-bold">{form.trestbps * form.thalach} mmHg&bull;bpm</span>
                <span className="text-[#8b949e] ml-3">Workload State: </span>
                <span className="text-white font-bold">{prognosis?.rpp_workload ?? "Hyperdynamic Demand"}</span>
              </div>
              <span className="text-[#f85149] font-bold animate-pulse">
                {prognosis?.early_warning ?? "Active trajectory forecasting..."}
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {(prognosis?.forecast || [
                { minute: "t+15m", projected_risk: 84.8, projected_st_dep: 2.7, ischemic_threat: "CRITICAL THREAT" },
                { minute: "t+30m", projected_risk: 86.5, projected_st_dep: 2.9, ischemic_threat: "CRITICAL THREAT" },
                { minute: "t+45m", projected_risk: 88.2, projected_st_dep: 3.1, ischemic_threat: "CRITICAL THREAT" },
                { minute: "t+60m", projected_risk: 90.1, projected_st_dep: 3.3, ischemic_threat: "CRITICAL THREAT" }
              ]).map((step: any, idx: number) => (
                <div key={idx} className="p-4 rounded-md bg-[#0d1117] border border-[#21262d] font-mono text-xs space-y-2">
                  <div className="flex justify-between items-center border-b border-[#21262d] pb-1.5">
                    <span className="text-[#58a6ff] font-bold text-sm">{step.minute}</span>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-[#f85149]/20 text-[#f85149] font-bold">{step.ischemic_threat}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#8b949e]">Predicted Risk:</span>
                    <span className="text-white font-bold text-sm">{step.projected_risk}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#8b949e]">ST Depression:</span>
                    <span className="text-[#f85149] font-semibold">{step.projected_st_dep} mm</span>
                  </div>
                  <div className="h-1.5 w-full bg-[#161b22] rounded-full overflow-hidden mt-1">
                    <div className="h-full bg-[#f85149]" style={{ width: `${step.projected_risk}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* WORKSPACE 3: AUTONOMOUS IV INFUSION PUMP */}
        {activeTab === "pump" && (
          <div className="bg-[#161b22] border border-[#30363d] rounded-lg p-5">
            <div className="flex items-center justify-between pb-3 border-b border-[#21262d] mb-4">
              <div className="flex items-center gap-2">
                <Gauge className="w-4 h-4 text-[#3fb950]" />
                <span className="text-xs font-mono uppercase tracking-wider text-white font-bold">
                  Autonomous Closed-Loop Pharmacodynamic IV Infusion Pump
                </span>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#3fb950]/15 text-[#3fb950] border border-[#3fb950]/30 font-bold">
                Smart Pump Protocol
              </span>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
              <div className="lg:col-span-6 p-4 rounded-md bg-[#0d1117] border border-[#21262d] space-y-4 font-mono text-xs">
                <div className="text-[11px] text-[#58a6ff] font-bold uppercase tracking-wider">Pump Controller Settings</div>
                <div>
                  <div className="flex justify-between text-[11px] mb-1">
                    <span className="text-[#8b949e]">Target Systolic BP (Endpoint):</span>
                    <span className="text-[#3fb950] font-bold">{targetBp} mmHg</span>
                  </div>
                  <input
                    type="range" min="100" max="150" step="5"
                    value={targetBp}
                    onChange={e => setTargetBp(parseFloat(e.target.value))}
                    className="w-full accent-[#3fb950] cursor-pointer"
                  />
                </div>
                <div className="flex justify-between py-2 border-t border-[#21262d]">
                  <span className="text-[#8b949e]">Patient Weight (Normalized):</span>
                  <span className="text-white font-bold">70.0 kg</span>
                </div>
                <button
                  onClick={handlePredict}
                  className="w-full py-2.5 rounded bg-[#238636] hover:bg-[#2ea043] text-white font-semibold text-xs transition"
                >
                  Recalculate Pharmacodynamic Infusion Rate
                </button>
              </div>

              <div className="lg:col-span-6 p-4 rounded-md bg-[#0d1117] border border-[#21262d] space-y-3 font-mono text-xs">
                <div className="text-[11px] text-[#3fb950] font-bold uppercase tracking-wider">Active Infusion Output</div>
                <div className="flex justify-between items-center py-2 border-b border-[#21262d]">
                  <span className="text-[#8b949e]">Selected Agent:</span>
                  <span className="text-white font-bold text-sm">{pumpData?.drug ?? "Nitroglycerin (NTG)"}</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-[#21262d]">
                  <span className="text-[#8b949e]">Volumetric Delivery:</span>
                  <span className="text-[#3fb950] font-black text-xl">{pumpData?.infusion_rate_ml_hr ?? "22.5"} mL/hr</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-[#21262d]">
                  <span className="text-[#8b949e]">Gravity Drip Equivalency:</span>
                  <span className="text-[#58a6ff] font-bold">{pumpData?.infusion_gtts_min ?? "22.5"} gtts/min</span>
                </div>
                <p className="text-[11px] text-[#8b949e] leading-relaxed pt-1">
                  <strong>Directive: </strong>{pumpData?.directive ?? "Titrate NTG to reduce cardiac afterload."}
                </p>
                <div className="p-2.5 rounded bg-[#f85149]/10 border border-[#f85149]/30 text-[10px] text-[#f85149]">
                  {pumpData?.safety_limit ?? "Cease infusion if SBP drops below 95 mmHg."}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* WORKSPACE 4: 3D MYOCARDIUM */}
        {activeTab === "ecg" && (
          <div className="bg-[#161b22] border border-[#30363d] rounded-lg p-4">
            <span className="text-xs font-mono uppercase tracking-wider text-white font-bold mb-3 flex items-center gap-2">
              <HeartPulse className="w-4 h-4 text-[#f85149]" />
              Real-Time Myocardial Cadence (WebGL)
            </span>
            <div className="h-[360px]">
              <HeartCanvas riskScore={isAsystole ? 100 : (result?.risk_score_percentage ?? 83.2)} />
            </div>
          </div>
        )}

        {/* WORKSPACE 5: SHAP */}
        {activeTab === "xai" && (
          <div className="bg-[#161b22] border border-[#30363d] rounded-lg p-5">
            <div className="flex items-center justify-between pb-3 border-b border-[#21262d] mb-4">
              <span className="text-xs font-mono uppercase tracking-wider text-white font-bold flex items-center gap-2">
                <Cpu className="w-4 h-4 text-[#58a6ff]" />
                SHAP Game-Theoretic Feature Weights
              </span>
            </div>
            <div className="space-y-3 font-mono text-xs">
              {(result?.shap_contributions || [
                { feature: "ca", impact: 0.1421 },
                { feature: "oldpeak", impact: 0.1189 },
                { feature: "thal", impact: 0.0865 },
                { feature: "cp", impact: 0.0759 },
                { feature: "trestbps", impact: 0.0412 }
              ]).map((item: any, idx: number) => (
                <div key={idx}>
                  <div className="flex justify-between text-[#8b949e] text-[11px] mb-1">
                    <span>{item.feature.toUpperCase()}</span>
                    <span className={item.impact >= 0 ? "text-[#f85149] font-bold" : "text-[#3fb950] font-bold"}>
                      {item.impact >= 0 ? `+${item.impact}` : item.impact}
                    </span>
                  </div>
                  <div className="h-2 w-full bg-[#0d1117] rounded-full overflow-hidden border border-[#21262d]">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        item.impact >= 0 ? "bg-[#f85149]" : "bg-[#3fb950]"
                      }`}
                      style={{ width: `${Math.min(Math.abs(item.impact) * 140, 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* WORKSPACE 6: BOARD & VOICE AI */}
        {activeTab === "board" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
            <div className="lg:col-span-7 bg-[#161b22] border border-[#30363d] rounded-lg overflow-hidden flex flex-col">
              <div className="bg-[#21262d] px-4 py-3 border-b border-[#30363d] flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Users className="w-4 h-4 text-[#58a6ff]" />
                  <span className="text-xs font-bold text-white uppercase tracking-wider font-mono">
                    Multi-Agent Clinical Consensus Quorum
                  </span>
                </div>
              </div>
              <div className="p-4 text-xs font-mono min-h-[300px] max-h-[460px] overflow-y-auto leading-relaxed bg-[#0d1117] text-[#c9d1d9] whitespace-pre-wrap">
                {boardStreamText || (
                  <span className="text-[#8b949e]">
                    Click "Execute Full Diagnostic Pipeline" in Matrix tab to stream live consensus between Dr. Cardiologist, Dr. Pharmacologist, Cardiac Dietitian, and Dr. Safety Auditor.
                  </span>
                )}
                {isBoardStreaming && <span className="inline-block w-2 h-3.5 ml-1 bg-[#58a6ff] animate-pulse" />}
              </div>
            </div>

            {/* Voice Copilot */}
            <div className="lg:col-span-5 bg-[#161b22] border border-[#30363d] rounded-lg overflow-hidden flex flex-col justify-between">
              <div>
                <div className="bg-[#21262d] px-4 py-3 border-b border-[#30363d] flex items-center justify-between">
                  <span className="text-xs font-bold text-[#58a6ff] font-mono flex items-center gap-2">
                    <Sparkles className="w-4 h-4" />
                    Hands-Free Voice AI Copilot
                  </span>
                  <button
                    onClick={toggleVoice}
                    className={`p-1.5 rounded transition ${
                      isListening ? "bg-[#f85149] text-white animate-pulse" : "bg-[#161b22] text-[#8b949e] hover:text-white"
                    }`}
                    title={isListening ? "Listening... Click to stop" : "Click to speak voice command"}
                  >
                    {isListening ? <Mic className="w-4 h-4" /> : <MicOff className="w-4 h-4" />}
                  </button>
                </div>

                <div className="p-3 max-h-[340px] overflow-y-auto space-y-2.5 text-xs">
                  {chatMessages.map((msg, i) => (
                    <div
                      key={i}
                      className={`p-3 rounded-md ${
                        msg.role === "assistant"
                          ? "bg-[#0d1117] border border-[#30363d] text-[#c9d1d9]"
                          : "bg-[#58a6ff]/10 border border-[#58a6ff]/30 text-white ml-6"
                      }`}
                    >
                      <div className="text-[10px] font-mono text-[#8b949e] mb-1 font-semibold flex items-center justify-between">
                        <span>{msg.role === "assistant" ? "CardioSense Core AI" : "Attending Physician (Voice/Text)"}</span>
                        {msg.role === "assistant" && (
                          <button onClick={() => speakVoice(msg.content)} className="hover:text-[#58a6ff]">
                            <Volume2 className="w-3.5 h-3.5" />
                          </button>
                        )}
                      </div>
                      <p className="leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                    </div>
                  ))}
                  <div ref={chatBottomRef} />
                </div>
              </div>

              <form onSubmit={handleSendMessage} className="p-3 border-t border-[#30363d] bg-[#0d1117] flex gap-2">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={e => setInputMessage(e.target.value)}
                  placeholder={isListening ? "Listening to your microphone..." : "Type or speak emergency order..."}
                  className="flex-1 bg-[#161b22] border border-[#30363d] rounded px-3 py-2 text-xs text-white placeholder-[#8b949e] focus:border-[#58a6ff] focus:outline-none font-mono"
                />
                <button
                  type="submit"
                  disabled={chatStreaming}
                  className="px-3.5 py-2 rounded bg-[#21262d] hover:bg-[#30363d] border border-[#30363d] text-[#58a6ff] text-xs font-semibold flex items-center gap-1 transition"
                >
                  {chatStreaming ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
                </button>
              </form>
            </div>
          </div>
        )}

        {/* WORKSPACE 7: RX & DIET */}
        {activeTab === "rx" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
            <div className="lg:col-span-6 bg-[#161b22] border border-[#30363d] rounded-lg p-5">
              <span className="text-xs font-mono uppercase tracking-wider text-white font-bold block mb-3">Clinical Prescription (Rx)</span>
              <div className="space-y-3 text-xs font-mono">
                <div className="p-3 rounded-md bg-[#0d1117] border border-[#21262d]">
                  <span className="text-[#58a6ff] font-bold block">1. Ecosprin 75 mg PO OD</span>
                  <span className="text-[#8b949e] text-[11px]">Antiplatelet protection.</span>
                </div>
                <div className="p-3 rounded-md bg-[#0d1117] border border-[#21262d]">
                  <span className="text-[#58a6ff] font-bold block">2. Atorvastatin 80 mg PO HS</span>
                  <span className="text-[#8b949e] text-[11px]">High-potency plaque stabilization.</span>
                </div>
              </div>
            </div>
            <div className="lg:col-span-6 bg-[#161b22] border border-[#30363d] rounded-lg p-5">
              <span className="text-xs font-mono uppercase tracking-wider text-white font-bold block mb-3">DASH Nutrition Chart</span>
              <div className="p-3 rounded-md bg-[#0d1117] border border-[#21262d] text-xs font-mono space-y-2">
                <p className="text-[#3fb950] font-bold">&check; Steel-cut oats (40g), Crushed flaxseeds (25g), Steamed spinach.</p>
                <p className="text-[#f85149] font-bold">&#10006; Zero palm oil, Trans-fats strictly banned, Sodium &lt; 1,500 mg/day.</p>
              </div>
            </div>
          </div>
        )}

        {/* WORKSPACE 8: COHORT */}
        {activeTab === "cohort" && (
          <div className="bg-[#161b22] border border-[#30363d] rounded-lg overflow-hidden p-5">
            <span className="text-xs font-mono uppercase tracking-wider text-white font-bold block mb-3">ICU Ward Cohort Table</span>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs font-mono">
                <thead>
                  <tr className="border-b border-[#21262d] bg-[#0d1117] text-[#8b949e]">
                    <th className="py-2.5 px-3">Bed</th>
                    <th className="py-2.5 px-3">Patient ID</th>
                    <th className="py-2.5 px-3">Demographics</th>
                    <th className="py-2.5 px-3">BP / HR</th>
                    <th className="py-2.5 px-3">Risk</th>
                    <th className="py-2.5 px-3">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#21262d]">
                  {cohort.map((pt, i) => (
                    <tr key={i}>
                      <td className="py-2.5 px-3 font-semibold text-white">{pt.bed}</td>
                      <td className="py-2.5 px-3 text-[#58a6ff]">{pt.patient_id}</td>
                      <td className="py-2.5 px-3">{pt.age}y / {pt.sex}</td>
                      <td className="py-2.5 px-3">{pt.bp} &bull; {pt.hr}</td>
                      <td className="py-2.5 px-3 font-bold text-white">{pt.risk_score}%</td>
                      <td className="py-2.5 px-3 text-[#8b949e]">{pt.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

      </div>

      <footer className="border-t border-[#21262d] py-5 text-center text-xs text-[#8b949e] mt-10 font-mono">
        CardioSense Autonomous Core &bull; 60m Prognosis Engine &bull; Autotitrated IV Pump &bull; Hands-Free Voice AI &bull; FHIR R4 Compliant
      </footer>
    </div>
  );
}
