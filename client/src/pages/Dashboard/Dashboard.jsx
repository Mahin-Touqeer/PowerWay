import { BrainCog, Rocket, Sun, Wind } from "lucide-react";
import React, { useState } from "react";
import ToggleButton from "./ToggleButton";
import ActiveUnitsSlider from "./ActiveUnitsSlider";
import getLatLong, {
  getLast1HourSolarData,
  getWindPastAndFutureData,
} from "@/api/geocodingapi.js";
// import { getForecastWeather } from "@/api/getForecastWeather";

/* ---------------- Reusable UI Components ---------------- */

const Card = ({ children, className = "" }) => (
  <div
    className={`bg-white dark:bg-background-dark border border-slate-200 dark:border-slate-800 rounded shadow-sm ${className}`}
  >
    {children}
  </div>
);

const StatRow = ({ label, value }) => (
  <div className="flex justify-between items-center p-3 rounded bg-white/5 border border-white/10">
    <span className="text-sm text-white">{label}</span>
    <span className="text-sm font-bold text-[#39DF79]">{value}</span>
  </div>
);

const OutputCard = ({
  title,
  value,
  unit,
  icon,
  accent = "text-primary",
  highlight = false,
}) => (
  <div
    className={`p-6 rounded-sm flex items-center justify-between border ${
      highlight
        ? "bg-primary text-background-dark border-primary/30 shadow-lg"
        : "bg-white dark:bg-background-dark border-slate-200 dark:border-slate-800"
    }`}
  >
    <div>
      <p
        className={`text-xs font-bold uppercase tracking-widest ${highlight ? "opacity-70" : "text-slate-400"}`}
      >
        {title}
      </p>
      <p
        className={`text-2xl font-black ${highlight ? "" : "text-slate-800 dark:text-white"}`}
      >
        {value} <span className="text-sm font-normal">{unit}</span>
      </p>
    </div>
    <span className={`material-symbols-outlined text-3xl ${accent}`}>
      {icon}
    </span>
  </div>
);

/* ---------------- Dashboard ---------------- */

export default function Dashboard() {
  const [isLoading, setIsLoading] = useState(false);
  const [location, setLocation] = useState(null);
  const [solarEfficiencyFactor, setSolarEfficiencyFactor] = useState(null);
  const [windEfficiencyFactor, setWindEfficiencyFactor] = useState(null);
  const [weatherData, setWeatherData] = useState({});
  const [windData, setWindData] = useState({
    maxCapacity: 120,
    availaible: true,
    totalUnits: 100,
    activeUnits: 50,
  });
  const [solarData, setSolarData] = useState({
    maxCapacity: 150,
    availaible: true,
    totalUnits: 100,
    activeUnits: 50,
  });

  async function handleSubmit() {
    setIsLoading(true);
    console.log("toSubmit: ", location);
    //fetch weatherData
    if (!location) {
      alert("Please enter location");
      return;
    }

    const { lat, lon } = await getLatLong(location);

    if (!lat || !lon) {
      alert("location invalid");
      return;
    }
    console.log(lat);
    console.log(lon);
    try {
      const data = await getLast1HourSolarData(lat, lon);
      // console.log(data);
      const { temp, windSpeed, humidity } = data[0];

      setWeatherData({
        temperature: temp.toFixed(2),
        windSpeed: windSpeed.toFixed(2),
        humidity,
      });

      const res = await fetch("http://localhost:8000/predict/solar", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ data }),
      });
      const { prediction } = await res.json();
      // console.log(json);
      setSolarEfficiencyFactor(prediction);
      setIsLoading(false);
      // for day Wind
      // const windPayload = await getWindPastAndFutureData(lat, lon);

      const windRes = await fetch("http://localhost:8000/predict/wind", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data }),
      });
      const { prediction: windPrediction } = await windRes.json();

      // const windJson = await windRes.json();
      setWindEfficiencyFactor(windPrediction);

      //SUM AUTH, MONs , winter
    } catch (e) {
      console.log(e);
    }

    // setWeatherData(data);
    // console.log(data);

    // console.log(solarData);
    // console.log(windData);

    //fetch efficiency factor
  }

  return (
    <div className="space-y-10 max-w-7xl mx-auto">
      {/* ========== AI Efficiency Engine ========== */}
      <section className="relative overflow-hidden rounded-lg aiEfficiency shadow-sm p-8 text-white">
        <div className="absolute -top-32 -right-32 w-64 h-64 bg-primary/20 blur-[120px]" />

        <div className="relative space-y-6">
          <h2 className="text-xl font-bold flex items-center gap-2">
            {/* <span className="material-symbols-outlined text-primary">
              psychology
            </span> */}
            <BrainCog color="#39DF79" size={28} />
            AI Efficiency Engine
          </h2>

          <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
            <div className="space-y-4">
              <p className="text-xs uppercase tracking-widest text-slate-400 font-bold">
                Weather Inputs
              </p>
              <StatRow
                label="Temperature"
                value={weatherData.temperature || "..."}
              />
              <StatRow
                label="Wind Speed"
                value={weatherData.windSpeed || "..."}
              />
              <StatRow label="Humidity" value={weatherData.humidity || "..."} />
            </div>

            <div className="lg:col-span-3 flex items-center justify-center border-2 border-dashed border-white/10 rounded bg-white/5 p-6">
              <div>
                <input
                  type="text"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  placeholder="Enter location (lat, lng or city)"
                  className="w-full mb-6 px-4 py-3 rounded-sm bg-white/10 border border-white/15 text-white placeholder:text-white/40 outline-none focus:border-[#39E079] focus:ring-1 focus:ring-[#39E079]/40 transition"
                />
                <button
                  className="bg-[#39E079] text-black px-8 py-4 rounded-sm font-bold flex items-center gap-3 shadow-lg shadow-primary/20 hover:scale-[1.01] transition duration-200 cursor-pointer w-sm"
                  onClick={handleSubmit}
                >
                  <span className="material-symbols-outlined">
                    <Rocket color="#212121" />
                  </span>
                  {isLoading ? "Running..." : "Run AI Efficiency Model"}
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>
      {/* ========== Asset Configuration ========== */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold">Asset Configuration</h2>
          <span className="text-xs px-2 py-1 rounded bg-slate-100 dark:bg-slate-800 text-slate-500">
            2 Assets Active
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Solar */}
          <Card className="p-6">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-6">
                <span className="bg-[#FFEDD5] p-2 rounded">
                  <Sun className="w-6 h-6" color="#EA580C" />
                </span>
                <div className="">
                  <h3 className="font-bold">Solar Array Config</h3>
                  <p className="text-xs text-slate-500">PV Efficiency & Load</p>
                </div>
              </div>
              <ToggleButton value={solarData} changeFunction={setSolarData} />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <Input
                label="Max Capacity (kW)"
                value={solarData.maxCapacity.toString()}
                onChange={(e) => {
                  // console.log(e.target.value);
                  setSolarData((prev) => ({
                    ...prev,
                    maxCapacity: Number(e.target.value),
                  }));
                  // setLocation(null);
                }}
              />
              <Input
                label="Total Units"
                value={solarData?.totalUnits?.toString()}
                onChange={(e) => {
                  // console.log(e.target.value);
                  setSolarData((prev) => ({
                    ...prev,
                    totalUnits: Number(e.target.value),
                  }));
                }}
              />

              <div className="col-span-2">
                <ActiveUnitsSlider
                  myvalue={solarData}
                  changeFunction={setSolarData}
                />
              </div>
            </div>
          </Card>

          {/* Wind */}
          <Card className="p-6">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-6">
                <span className="bg-[#DBEAFE] p-2 rounded">
                  <Wind className="w-6 h-6" color="#115AEB" />
                </span>
                <div className="">
                  <h3 className="font-bold">Wind farm config</h3>
                  <p className="text-xs text-slate-500">
                    Turbine specifications
                  </p>
                </div>
              </div>

              <ToggleButton value={windData} changeFunction={setWindData} />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <Input
                label="Max Capacity (kW)"
                value={windData.maxCapacity.toString()}
                onChange={(e) =>
                  setWindData((prev) => ({
                    ...prev,
                    maxCapacity: Number(e.target.value),
                  }))
                }
              />
              <Input
                label="Total Units"
                value={windData.totalUnits.toString()}
                onChange={(e) =>
                  setWindData((prev) => ({
                    ...prev,
                    totalUnits: Number(e.target.value),
                  }))
                }
              />

              <div className="col-span-2">
                <ActiveUnitsSlider
                  myvalue={windData}
                  changeFunction={setWindData}
                />
              </div>
            </div>
          </Card>
        </div>
      </section>
      {/* ========== Power Output Calculation ========== */}
      <section>
        <h2 className="text-xl font-bold mb-6">Power Output Calculation</h2>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="lg:col-span-2 p-6 space-y-6">
            <Formula
              title="Solar Formula"
              formula="P = ef × maxCapacity × cf × availaiblility"
              result={
                solarEfficiencyFactor
                  ? (solarEfficiencyFactor *
                      solarData.activeUnits *
                      solarData.availaible *
                      solarData.maxCapacity) /
                    solarData.totalUnits
                  : "..."
              }
            />
            <Formula
              title="Wind Formula"
              formula="P = ef × maxCapacity × cf × availaiblility"
              result={windEfficiencyFactor ? windEfficiencyFactor : "..."}
            />
          </Card>

          <div className="space-y-4">
            <OutputCard
              title="Solar Output"
              value={
                solarEfficiencyFactor
                  ? (solarEfficiencyFactor *
                      solarData.activeUnits *
                      solarData.availaible *
                      solarData.maxCapacity) /
                    solarData.totalUnits
                  : "..."
              }
              unit="kW"
              icon={<Sun fill="orange" size={32} />}
              accent="text-orange-500"
            />
            <OutputCard
              title="Wind Output"
              value={windEfficiencyFactor ? windEfficiencyFactor : "..."}
              unit="kW"
              icon={<Wind color="#115AEB" size={32} />}
              accent="text-blue-500"
            />
            {/* <OutputCard
              title="Total Platform Output"
              value="174.69"
              unit="kW"
              icon="energy_savings_leaf"
              highlight
            /> */}
          </div>
        </div>
      </section>
    </div>
  );
}

/* ---------------- Small Helpers ---------------- */

const Input = ({ label, value, onChange }) => (
  <div className="space-y-1">
    <label className="text-xs uppercase font-semibold text-slate-500">
      {label}
    </label>
    <input
      type="number"
      value={value}
      onChange={onChange}
      // readOnly
      className="w-full rounded-lg p-2.5 text-sm bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800"
    />
  </div>
);

const Formula = ({ title, formula, result }) => (
  <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-4 font-mono">
    <div className="flex justify-between mb-2 border-b border-slate-200 dark:border-slate-800 pb-2">
      <span className="text-xs font-bold uppercase text-primary">{title}</span>
      <code className="text-xs">{formula}</code>
    </div>
    <p className="text-sm font-medium">
      Result = <span className="text-primary font-bold">{result}</span>
    </p>
  </div>
);
