import React, { useState } from "react";
import {
  Brain,
  Info,
  CheckCircle,
  TrendingUp,
  BarChart3,
  Lightbulb,
  PlusCircle,
  BrainCog,
} from "lucide-react";
import getLatLong, { buildLoadPredictionPayload } from "@/api/geocodingapi";

/* ---------------- Reusable Components ---------------- */

const Card = ({ children, className = "" }) => (
  <div
    className={`bg-white dark:bg-[#1a2e21] border border-[#e8f2ec] dark:border-[#1e3226] rounded-xl shadow-sm ${className}`}
  >
    {children}
  </div>
);

const InputField = ({ label, value, onChange, type = "number" }) => (
  <label className="flex flex-col flex-1 min-w-[180px]">
    <div className="flex items-center gap-2 mb-2">
      <span className="text-sm font-medium">{label}</span>
      <Info size={14} className="text-primary/60" />
    </div>
    <input
      type={type}
      value={value}
      onChange={onChange}
      className="h-14 rounded-sm px-4 text-lg font-medium bg-white/5 border border-primary/20 focus:ring-1  focus:border-[#39DF79] text-white"
    />
  </label>
);

const MetricBox = ({ label, value, suffix = "", highlight = false }) => (
  <div className="p-4 rounded-lg bg-background-light dark:bg-[#122017] transition border hover:border-primary/20">
    <p className="text-xs text-gray-500 mb-1">{label}</p>
    <p className={`text-xl font-bold ${highlight ? "text-primary" : ""}`}>
      {value} {suffix}
    </p>
  </div>
);

/* ---------------- Load Page ---------------- */

export default function Load() {
  const [energy, setEnergy] = useState(1450);
  const [price, setPrice] = useState(8.2);
  const [GDP, setGDP] = useState(925);
  const [location, setLocation] = useState(null);
  const [load, setLoad] = useState(null);
  const [lag, setLag] = [[1, 2, 3, 4, 5]];
  const [isLoading, setIsLoading] = useState(false);

  async function handleCalculate() {
    setIsLoading(true);
    console.log("toSubmit: ", location);
    //fetch weatherData
    if (!location || !energy || !price || !GDP) {
      alert("Please enter all fields");
      return;
    }

    const { lat, lon } = await getLatLong(location);

    if (!lat || !lon) {
      alert("location invalid");
      return;
    }
    try {
      const data = await buildLoadPredictionPayload(lat, lon);
      console.log(data);

      const payload = {
        ...data,
        GDP,
        PerCapitaEnergyUse: energy,
        ElectricityPrice: price * 3.38,
        lag_1: lag[0],
        lag_2: lag[1],
        lag_3: lag[2],
        lag_4: lag[3],
        lag_5: lag[4],
      };

      console.log(payload);

      const res = await fetch(
        "https://compilation-instrument-programme-imperial.trycloudflare.com/predict",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        },
      );
      // console.log(res);
      const { predicted_load_kw } = await res.json();
      // console.log(x);
      setLoad(predicted_load_kw);
      console.log(typeof predicted_load_kw);

      setIsLoading(false);

      // const { prediction } = await res.json();
      // // console.log(json);
      // setSolarEfficiencyFactor(prediction);
      //SUM AUTH, MONs , winter
    } catch (e) {
      console.log(e);
    }
    // const res = await fetch("http://localhost:8000/predict/load", {
    //   method: "POST",
    //   headers: {
    //     "Content-Type": "application/json",
    //   },
    //   body: JSON.stringify({ energy, price, location }),
    // });
    // const { prediction } = await res.json();
    // setLoad(prediction);
    // setIsLoading(false);
  }

  return (
    <div className="max-w-7xl mx-auto space-y-16 ">
      {/* ===== AI Load Engine ===== */}
      <section className="bg-gradient-to-br from-[#0e1a13] to-[#1e3c2b] rounded-lg p-8 text-white shadow-2xl">
        <div className="flex items-center gap-3 mb-3">
          <BrainCog color="#39DF79" size={28} />
          <h1 className="text-xl font-bold">AI Load Estimation Engine</h1>
        </div>

        <p className="text-[#39DF79] text-lg mb-8 max-w-2xl">
          Derive real-time electrical load using consumption behavior and
          pricing inputs.
        </p>

        <div className="flex flex-wrap items-end gap-6">
          <InputField
            label="Per Capita Energy Use (kWh)"
            value={energy}
            onChange={(e) => setEnergy(e.target.value)}
          />
          <InputField
            label="Electricity Price (INR/kWh)"
            value={price}
            onChange={(e) => setPrice(e.target.value)}
          />
          <InputField
            label="GDP"
            value={GDP}
            type="text"
            onChange={(e) => setGDP(e.target.value)}
          />
          <InputField
            label="Location (City)"
            value={location}
            type="text"
            onChange={(e) => setLocation(e.target.value)}
          />

          <button
            onClick={handleCalculate}
            className="h-14 px-8 rounded-sm bg-[#39DF79] text-[#0e1a13] font-bold text-lg shadow-lg shadow-primary/30 hover:opacity-90 transition"
          >
            {isLoading ? "Calculating..." : "Calculate Load"}
          </button>
        </div>
      </section>

      {/* ===== Derived Load ===== */}

      {load ? (
        <section className="flex flex-col items-center text-center">
          <p className="italic text-gray-500 mb-6">
            Estimated Load (kW) = f(Consumption, Price)
          </p>

          <div className="relative">
            <div className="absolute -inset-6 bg-primary/10 blur-xl rounded-full" />
            <div className="relative">
              <p className="text-[#39DF79] text-sm font-bold uppercase tracking-widest">
                Derived Load
              </p>
              <div className="flex items-baseline gap-2">
                <span className="text-7xl font-black font-mono">
                  {load.toString().split(".")[0]}.
                  {load.toString().split(".")[1].slice(0, 2)}
                </span>
                <span className="text-2xl text-gray-400">kW</span>
              </div>
              <div className="mt-4 inline-flex items-center gap-1 px-4 py-1 rounded-full bg-[#39DF79] text-primary text-xs font-bold uppercase">
                <CheckCircle size={14} />
                Optimized Estimate
              </div>
            </div>
          </div>
        </section>
      ) : (
        <section className="flex flex-col items-center justify-center text-center">
          {/* Glow */}
          <div className="relative ">
            <div className="absolute -inset-8 bg-primary/10 blur-2xl rounded-full" />
            <div className="relative w-24 h-24 rounded-full border-2 border-dashed border-primary/40 flex items-center justify-center">
              <PlusCircle className="text-primary" size={40} />
            </div>
          </div>

          <h3 className="text-xl font-bold mb-2">No Load Calculated Yet</h3>

          <p className="text-gray-500 max-w-md mb-6">
            Enter per-capita energy usage and electricity price, then run the AI
            Load Estimation Engine to derive an optimized load value.
          </p>

          <div className="flex items-center gap-2 text-sm text-primary font-semibold">
            <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            Waiting for input values
          </div>
        </section>
      )}

      {/* ===== Lower Section ===== */}
    </div>
  );
}

///
/* ---------------- Helpers ---------------- */

const Benchmark = ({ label, value, width, highlight }) => (
  <div>
    <div className="flex justify-between text-sm mb-1">
      <span>{label}</span>
      <span className={`font-bold ${highlight ? "text-primary" : ""}`}>
        {value}
      </span>
    </div>
    <div className="h-1.5 bg-gray-100 dark:bg-[#122017] rounded-full">
      <div
        className={`h-full rounded-full ${highlight ? "bg-primary" : "bg-gray-400"}`}
        style={{ width }}
      />
    </div>
  </div>
);
