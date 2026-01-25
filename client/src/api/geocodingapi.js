import { fetchWeatherApi } from "openmeteo";

export async function getLast1HourSolarData(lat, lon) {
  const params = {
    latitude: lat,
    longitude: lon,
    hourly: [
      "shortwave_radiation", // GHI
      "direct_normal_irradiance", // DNI
      "diffuse_radiation", // DHI
      "temperature_2m",
      "wind_speed_10m",
      "relative_humidity_2m",
    ],
    timezone: "auto",
  };

  const url = "https://api.open-meteo.com/v1/forecast";
  const responses = await fetchWeatherApi(url, params);
  const response = responses[0];
  const hourly = response.hourly();

  const utcOffsetSeconds = response.utcOffsetSeconds();
  const now = new Date();

  // Day & season (same logic as your Solcast version)
  const day = now.getDay() === 0 ? 7 : now.getDay();
  const season = getSeason(now.getMonth() + 1);

  // Take the LAST hourly index
  const lastIdx = hourly.variables(0).valuesArray().length - 1;

  const baseTimestamp =
    (Number(hourly.time()) + lastIdx * hourly.interval() + utcOffsetSeconds) *
    1000;

  // Hourly base values
  const base = {
    GHI: hourly.variables(0).valuesArray()[lastIdx],
    DNI: hourly.variables(1).valuesArray()[lastIdx],
    DHI: hourly.variables(2).valuesArray()[lastIdx],
    temp: hourly.variables(3).valuesArray()[lastIdx],
    windSpeed: hourly.variables(4).valuesArray()[lastIdx],
    humidity: hourly.variables(5).valuesArray()[lastIdx],
  };

  // Interpolate into 12 × 5-minute points
  const intervalMs = 5 * 60 * 1000;

  const processed = Array.from({ length: 18 }, (_, i) => ({
    timestamp: new Date(baseTimestamp - (11 - i) * intervalMs).toISOString(),

    season,
    day,

    GHI: base.GHI,
    DNI: base.DNI,
    DHI: base.DHI,

    temp: base.temp,
    windSpeed: base.windSpeed,
    humidity: base.humidity,
  }));

  return processed;
}

export default async function getLatLong(place) {
  const res = await fetch(
    `https://geocoding-api.open-meteo.com/v1/search?name=${place}&count=1`,
  );

  const data = await res.json();

  if (!data.results) return null;

  return {
    lat: data.results[0].latitude,
    lon: data.results[0].longitude,
  };
}

//SUM AUTH, MONs , winter
function getSeason(month) {
  if ([12, 1, 2].includes(month)) return 1;
  if ([3, 4, 5].includes(month)) return 2;
  if ([6, 7, 8].includes(month)) return 3;
  if ([9, 10, 11].includes(month)) return 4;
  return "post-monsoon";
}

export async function buildLoadPredictionPayload(lat, lon) {
  const params = {
    latitude: lat,
    longitude: lon,
    hourly: [
      "temperature_2m",
      "relative_humidity_2m",
      "wind_speed_10m",
      "rain",
      "shortwave_radiation",
    ],
    timezone: "auto",
  };

  const url = "https://api.open-meteo.com/v1/forecast";
  const responses = await fetchWeatherApi(url, params);
  const response = responses[0];
  const hourly = response.hourly();

  const now = new Date();

  // ---- time features ----
  const timestamp = now.toISOString();
  const hour = now.getHours();
  const dayOfWeek = now.getDay() === 0 ? 7 : now.getDay(); // 1–7
  const month = now.getMonth() + 1;

  // ---- take latest available hourly index ----
  const lastIdx = hourly.variables(0).valuesArray().length - 1;

  const temperature = hourly.variables(0).valuesArray()[lastIdx];
  const humidity = hourly.variables(1).valuesArray()[lastIdx];
  const windSpeed = hourly.variables(2).valuesArray()[lastIdx];
  const rainfall = hourly.variables(3).valuesArray()[lastIdx];
  const solarIrradiance = hourly.variables(4).valuesArray()[lastIdx];

  // ---- final payload (matches FastAPI exactly) ----
  return {
    Timestamp: formatTimestampForBackend(new Date()),

    Temperature: temperature,
    Humidity: humidity,
    WindSpeed: windSpeed,
    Rainfall: rainfall,
    SolarIrradiance: solarIrradiance,

    // ---- optional socio-economic placeholders ----
    // GDP: 0  ,
    // "Per Capita Energy Use (kWh)": 0,
    // "Electricity Price (LKR/kWh)": 0,

    DayOfWeek: dayOfWeek,
    HourOfDay: hour,
    Month: month,
    PublicEvent: 0,

    // ---- lag placeholders (replace later with real history) ----
    lag_1: 0,
    lag_2: 0,
    lag_3: 0,
    lag_4: 0,
    lag_5: 0,
  };
}
function formatTimestampForBackend(date = new Date()) {
  const yyyy = date.getFullYear();
  const mm = String(date.getMonth() + 1).padStart(2, "0");
  const dd = String(date.getDate()).padStart(2, "0");

  const hh = String(date.getHours()).padStart(2, "0");
  const min = String(date.getMinutes()).padStart(2, "0");
  const ss = String(date.getSeconds()).padStart(2, "0");

  return `${yyyy}-${mm}-${dd} ${hh}:${min}:${ss}`;
}

export async function getWindPastAndFutureData(lat, lon) {
  const params = {
    latitude: lat,
    longitude: lon,
    hourly: ["wind_speed_10m", "temperature_2m", "relative_humidity_2m"],
    past_days: 1,
    forecast_days: 1,
    timezone: "auto",
  };

  const url = "https://api.open-meteo.com/v1/forecast";
  const responses = await fetchWeatherApi(url, params);
  const response = responses[0];

  const hourly = response.hourly();
  const utcOffsetSeconds = response.utcOffsetSeconds();

  const times = hourly.time();
  const interval = hourly.interval(); // 3600

  const wind = hourly.variables(0).valuesArray();
  const temp = hourly.variables(1).valuesArray();
  const hum = hourly.variables(2).valuesArray();

  // =====================================================
  // 🔹 SPLIT PAST + FUTURE
  // =====================================================

  const now = new Date();

  const pastHourly = [];
  const futureHourly = [];

  for (let i = 0; i < wind.length; i++) {
    const t = new Date(
      (Number(times) + i * interval + utcOffsetSeconds) * 1000,
    );

    const point = {
      Time: t.toISOString(),
      Wind_speed: wind[i],
      Temperature: temp[i],
      Humidity: hum[i],
    };

    if (t <= now) pastHourly.push(point);
    else futureHourly.push(point);
  }

  // take exactly last 24 hourly points
  const past24h = pastHourly.slice(-24);
  const future24h = futureHourly.slice(0, 24);

  // =====================================================
  // 🔹 INTERPOLATE HOURLY → 5 MIN
  // =====================================================

  const past_5min_data = [];

  for (let h = 0; h < past24h.length - 1; h++) {
    const p1 = past24h[h];
    const p2 = past24h[h + 1];

    const t1 = new Date(p1.Time).getTime();
    const t2 = new Date(p2.Time).getTime();

    for (let i = 0; i < 12; i++) {
      const ratio = i / 12;

      past_5min_data.push({
        Time: formatTimestampForBackend(),

        Wind_speed: p1.Wind_speed + ratio * (p2.Wind_speed - p1.Wind_speed),

        Temperature: p1.Temperature + ratio * (p2.Temperature - p1.Temperature),

        Humidity: p1.Humidity + ratio * (p2.Humidity - p1.Humidity),
      });
    }
  }

  // ensure exactly 288 points
  const pastTrimmed = past_5min_data.slice(-288);

  return {
    past_5min_data: pastTrimmed,
    future_hourly_data: future24h.map((p) => ({
      Time: p.Time,
      Wind_speed: p.Wind_speed,
      Temperature: p.Temperature,
    })),
  };
}
