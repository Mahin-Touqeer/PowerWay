const { VITE_BACKEND_URL } = import.meta.env;
export async function getForecastWeather(lat, lon) {
  console.log(`Backend url: ${VITE_BACKEND_URL}`);

  const weatherData = await fetch(
    `${VITE_BACKEND_URL}/getweatherdata?lat=${lat}&lon=${lon}`,
  );
  console.log(weatherData);
  const data = await weatherData.json();
  return data;
}
