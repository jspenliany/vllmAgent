from langchain_core.tools import tool
from datetime import datetime
from chttps.openGeographicApi import get_location_data
from chttps.openWeatherApi import callWeatherAPI



@tool
def get_system_time():
    """Get the current local system time to synchronize schedules."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def get_coordinates(city: str):
    """Get the latitude and longitude for a given city name."""
    print(f"Getting coordinates for {city}")
    # get_location_data(city)
    # Mock logic - Replace with real Geocoding API call
    return get_location_data(city)

@tool
def get_weather(latitude: float, longitude: float, start_date: str, end_date: str):
    """Get weather data based on coordinates and date range (YYYY-MM-DD)."""
    print(f"Getting weather for {latitude}, {longitude}")
    # Mock logic - Replace with Open-Meteo or OpenWeatherMap API
    return callWeatherAPI(str(latitude), str(longitude), start_date, end_date)

tools = [get_coordinates, get_weather, get_system_time]