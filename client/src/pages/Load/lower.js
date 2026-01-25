<section className="grid grid-cols-1 lg:grid-cols-3 gap-8">
  {/* ---- Left ---- */}
  <div className="lg:col-span-2 space-y-6">
    <h3 className="text-xl font-bold">Load Configuration & Usage Context</h3>

    <Card className="p-6">
      <div className="flex justify-between mb-6">
        <div>
          <p className="text-sm uppercase font-bold text-gray-500">
            Derived Load Summary
          </p>
          <p className="text-xs text-gray-400 mt-1">
            Last updated: Oct 24, 2023 at 14:32
          </p>
        </div>
        <BarChart3 className="text-primary" />
      </div>

      <div className="grid grid-cols-3 gap-4">
        <MetricBox label="Energy Baseline" value="1,450" suffix="kWh" />
        <MetricBox label="Tariff Rate" value="₹8.20" suffix="/kWh" />
        <MetricBox label="Peak Intensity" value="High" highlight />
      </div>
    </Card>

    <Card className="p-6">
      <div className="flex justify-between items-center mb-8">
        <h4 className="font-bold flex items-center gap-2">
          <TrendingUp className="text-primary" />
          Consumption Sensitivity
        </h4>
      </div>

      <div className="space-y-8">
        {/* Usage */}
        <div>
          <div className="flex justify-between text-sm font-medium mb-2">
            <span>Usage Impact (kWh)</span>
            <span className="text-primary">+12%</span>
          </div>
          <div className="h-4 bg-gray-100 dark:bg-[#122017] rounded-full">
            <div className="h-full w-[72%] bg-primary rounded-full" />
          </div>
        </div>

        {/* Price */}
        <div>
          <div className="flex justify-between text-sm font-medium mb-2">
            <span>Price Elasticity (INR)</span>
            <span className="text-red-400">-5.2%</span>
          </div>
          <div className="h-4 bg-gray-100 dark:bg-[#122017] rounded-full relative">
            <div className="h-full w-[45%] bg-primary/40 rounded-full" />
            <div className="absolute left-[45%] top-0 h-full w-[1px] bg-red-400" />
          </div>
        </div>
      </div>
    </Card>
  </div>

  {/* ---- Right ---- */}
  <div className="space-y-6">
    <h3 className="text-xl font-bold">Insights</h3>

    <Card className="p-6 bg-primary/5 border-primary/10">
      <Lightbulb className="text-primary mb-3" size={28} />
      <h5 className="font-bold mb-2">Efficiency Tip</h5>
      <p className="text-sm text-gray-600 dark:text-gray-300">
        Increasing your base per-capita efficiency by 5% could reduce your peak
        load requirement by 1.2 kW.
      </p>
    </Card>

    <Card className="overflow-hidden">
      <div className="p-4 bg-background-light dark:bg-[#253d2f] border-b">
        <p className="text-xs uppercase font-bold text-gray-500">
          Regional Benchmark
        </p>
      </div>

      <div className="p-6 space-y-4">
        <Benchmark label="Global Average" value="14.2 kW" width="85%" />
        <Benchmark
          label="Your Location"
          value="12.4 kW"
          width="74%"
          highlight
        />
      </div>
    </Card>

    <div className="border-2 border-dashed border-gray-200 dark:border-gray-800 rounded-xl p-8 text-center text-gray-400">
      <PlusCircle size={36} className="mx-auto mb-2" />
      <p className="text-sm font-medium">
        Add another <br /> Comparison Input
      </p>
    </div>
  </div>
</section>;
