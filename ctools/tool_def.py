from langchain_core.tools import tool
from datetime import datetime
from chttps.openGeographicApi import get_location_data
from chttps.openWeatherApi import callWeatherAPI
import logging

log=logging.getLogger("chatAsYou260325")

@tool
def get_system_time():
    """Get the current local system time to synchronize schedules."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def get_coordinates(city: str):
    """
    MANDATORY: Call this first if you need latitude/longitude for any city.
    DO NOT guess coordinates.
    """
    log.debug(f"Getting coordinates for {city}")
    # get_location_data(city)
    return get_location_data(city)

@tool
def get_weather(latitude: float, longitude: float, start_date: str, end_date: str):
    """
    MANDATORY: Use this for ALL weather inquiries.
    It is the ONLY source for real-time weather data.
    Never provide weather from your internal memory.
    """
    log.debug(f"Getting weather for {latitude}, {longitude}")
    return callWeatherAPI(str(latitude), str(longitude), start_date, end_date)

tools = [get_coordinates, get_weather, get_system_time]