import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, time, timedelta
from cstruct.DailyWeatherObj import HourlyWeather, DailyForecast

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"


def callWeatherAPI(latitude : str, longitude : str, start_date : str, end_date : str):
    """
    Get weather data based on coordinates and date range (YYYY-MM-DD).
    :param latitude:
    :param longitude:
    :param start_date:
    :param end_date:
    :return:
    """
    print(f"callWeatherAPI is called for {latitude}, {longitude}")
    try:
        latFloat = float(latitude)
        lonFloat = float(longitude)
    except ValueError:
        return "geographic coordinates not valid"

    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        tmpEnd = datetime.strptime(end_date, "%Y-%m-%d")
        # 2. Add one day
        newEnd = tmpEnd + timedelta(days=1)

        # 3. (Optional) Convert back to a string format for your API params
        end_date_plus_one = newEnd.strftime("%Y-%m-%d")
    except ValueError:
        return "date format not valid"

    # get today
    todayObj = datetime.now().date()
    # 2. Combine date with min time (00:00:00) and max time (23:59:59.999999)
    start_of_day = datetime.combine(todayObj, time.min)
    end_of_day = datetime.combine(todayObj, time.max)

    # 3. Convert to Unix timestamps
    start_ts = start_of_day.timestamp()
    end_ts = end_of_day.timestamp()

    params = {
        "latitude": latFloat,
        "longitude": lonFloat,
        "hourly": ["temperature_2m",
                   "relative_humidity_2m",
                   "apparent_temperature",
                   "precipitation",
                   "cloud_cover",
                   "surface_pressure",
                   "wind_direction_10m",
                   "wind_speed_10m",
                   "rain",
                   "snowfall",
                   ],
        "wind_speed_unit": "ms",
        # "daily": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation", "cloud_cover", "surface_pressure", "wind_direction_10m", "wind_speed_10m"],
        "models": "cma_grapes_global",
        "start_date": start_date,
        "end_date": end_date_plus_one,
    }
    responses = openmeteo.weather_api(url, params=params)
    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    # print(f"Elevation: {response.Elevation()} m asl")
    # print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    timeStart = hourly.Time()
    timeEnd = hourly.TimeEnd()

    # Difference in hours gives us the starting index
    start_index = int((start_ts - timeStart) / 3600)
    end_index = start_index + 24  # 24 hours for the full day
    verifyStartHour = timeStart + start_index * 3600
    verifyStartHour = datetime.fromtimestamp(verifyStartHour)

    startDate = datetime.fromtimestamp(timeStart)
    endDate = datetime.fromtimestamp(timeEnd)
    # print(f"Start time: {timeStart} to {timeEnd}.....Date range: {startDate} to {endDate}...now is {todayObj}...start_index is {start_index}....verifyStartHour={verifyStartHour}")

    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_apparent_temperature = hourly.Variables(2).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(3).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(4).ValuesAsNumpy()
    hourly_surface_pressure = hourly.Variables(5).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(6).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(7).ValuesAsNumpy()
    hourly_rain = hourly.Variables(8).ValuesAsNumpy()
    hourly_snowfall = hourly.Variables(9).ValuesAsNumpy()
    # daily = response.Daily()
    # daily_temperature_2m = daily.Variables(0).ValuesAsNumpy()
    # daily_relative_humidity_2m = daily.Variables(1).ValuesAsNumpy()
    # daily_apparent_temperature = daily.Variables(2).ValuesAsNumpy()
    # daily_precipitation = daily.Variables(3).ValuesAsNumpy()
    # daily_cloud_cover = daily.Variables(4).ValuesAsNumpy()
    # daily_surface_pressure = daily.Variables(5).ValuesAsNumpy()
    # daily_wind_direction_10m = daily.Variables(6).ValuesAsNumpy()
    # daily_wind_speed_10m = daily.Variables(7).ValuesAsNumpy()

    # 1. Convert your starting timestamp into a datetime object
    start_base = datetime.fromtimestamp(start_ts)
    today_list = []
    for i in range(24):
        # 2. Add 'i' hours to the starting datetime
        current_hour_base = start_base + timedelta(hours=i)
        start_hour = current_hour_base.strftime("%Y-%m-%d %H:%M:%S")
        # print(f"i={i}")
        today_list.append(
            HourlyWeather(
                time=start_hour,
                temperature=hourly_temperature_2m[start_index + i],
                relative_humidity=hourly_relative_humidity_2m[start_index + i],
                apparent_temperature=hourly_apparent_temperature[start_index + i],
                precipitation=hourly_precipitation[start_index + i],
                cloud_cover=hourly_cloud_cover[start_index + i],
                surface_pressure=hourly_surface_pressure[start_index + i],
                wind_direction=hourly_wind_direction_10m[start_index + i],
                wind_speed=hourly_wind_speed_10m[start_index + i],
                rain=hourly_rain[start_index + i],
                snowfall=hourly_snowfall[start_index + i],
            )
        )
    # print("\n---------------\n")
    # print(today_list)
    # print("\n---------------\n")
    return today_list



if __name__ == '__main__':
    result = callWeatherAPI("68.18","19.16","2026-03-13","2026-03-15")
    print(result)