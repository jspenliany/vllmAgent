import requests

def get_location_data(city_name, count=1, country_code=None):
    """
    Fetches geographic information for a city name using Open-Meteo.
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": city_name,
        "count": count,
        "format": "json",
        "language": "zh",
    }

    # Optional filter by country code (e.g., 'US', 'GB')
    if country_code:
        params["countryCode"] = country_code

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()

        if "results" in data:
            # Returns a list of matching locations
            tmpData = data["results"]
            resultStr = f'{tmpData[0].get("latitude")}, {tmpData[0].get("longitude")}'
            return resultStr
        else:
            print(f"No results found for '{city_name}'.")
            return "37.77493, -122.41942"

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return "37.77493, -122.41942"


if __name__ == "__main__":
    geoLocation = get_location_data(city_name="San Francisco", country_code="US")
    print(geoLocation)