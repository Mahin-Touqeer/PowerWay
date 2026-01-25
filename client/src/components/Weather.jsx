import { getForecastWeather } from "../api/getForecastWeather";

function Weather() {
  async function handleSubmit(e) {
    e.preventDefault();
    const lat = Number(e.target[0].value);
    const lng = Number(e.target[1].value);

    console.log(lat);
    console.log(lng);

    const weatherData = await getForecastWeather(lat, lng);
    console.log(weatherData);
  }

  return (
    <div>
      <form
        onSubmit={handleSubmit}
        className="w-full h-full flex justify-center items-center"
      >
        <div>
          <input type="text" placeholder="lattitude" />
          <br />
          <input type="text" placeholder="longitude" />
          <br />
          <button type="submit">Submit</button>
        </div>
      </form>
    </div>
  );
}

export default Weather;
