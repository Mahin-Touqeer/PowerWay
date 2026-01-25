import { useState } from "react";

export default function ActiveUnitsSlider({ myvalue, changeFunction }) {
  const max = 48;
  const [value, setValue] = useState(42);

  const percentage = (myvalue?.activeUnits / myvalue?.totalUnits) * 100;
  const curtailmentFactor = (
    myvalue?.activeUnits / myvalue?.totalUnits
  ).toFixed(3);

  return (
    <div className="w-full max-w-lg">
      {/* Header */}
      <div className="flex justify-between text-xs font-semibold mb-2">
        <span className="text-slate-500 uppercase tracking-wide">
          Active Units
        </span>
        <span className="text-emerald-500 font-bold">
          {myvalue?.activeUnits} / {myvalue?.totalUnits}
        </span>
      </div>

      {/* Slider */}
      <div className="relative">
        {/* Track */}
        <div className="h-2 bg-slate-200 rounded-full" />

        {/* Filled Progress */}
        {/* <div
          className="absolute top-0 h-2 bg-emerald-400 rounded-full"
          style={{ width: `${percentage}%` }}
        /> */}

        {/* Thumb */}
        <div
          className="absolute top-1/2 w-4 h-4 bg-[#39E079] rounded-full shadow-md -translate-y-1/2"
          style={{ left: `calc(${percentage}% - 10px)` }}
        />

        {/* Hidden Native Range */}
        <input
          type="range"
          min={0}
          max={myvalue.totalUnits}
          value={myvalue.activeUnits}
          onChange={(e) =>
            changeFunction((prev) => ({
              ...prev,
              activeUnits: Number(e.target.value),
            }))
          }
          className="absolute inset-0 opacity-0 cursor-pointer"
        />
      </div>

      {/* Footer Text */}
      <p className="text-[11px] text-slate-400 mt-2">
        Curtailment Factor:{" "}
        <span className="font-medium">{curtailmentFactor}</span>
      </p>
    </div>
  );
}
