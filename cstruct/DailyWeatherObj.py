from dataclasses import dataclass
from typing import List

@dataclass
class HourlyWeather:
    time: str                # e.g., "2026-03-13T08:00"
    temperature: float       # °C
    relative_humidity: float # %
    apparent_temperature: float # Feel-like °C
    precipitation: float     # mm
    cloud_cover: float       # %
    surface_pressure: float  # hPa
    wind_direction: float    # degrees
    wind_speed: float        # km/h or m/s
    rain: float              # %
    snowfall: float          # %
@dataclass
class DailyForecast:
    date: str
    max_temp: float
    min_temp: float
    hourly_data: List[HourlyWeather]
