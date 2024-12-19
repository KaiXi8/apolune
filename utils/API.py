import requests
import json

def api_get(body: str = None):
    """
    Gets data from JPL 3 body data base API

    Args:
        body (dict): parameters to request

    Returns:
        periodic_orbits_data (dict): Queried orbits and associated data

    """
    url = "https://ssd-api.jpl.nasa.gov/periodic_orbits.api"
    
    response = requests.get(url, params=body)
    
    if response.status_code == 200:
        periodic_orbits_data = response.json()
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return None    
    
    if periodic_orbits_data:
        return periodic_orbits_data
    else:
        print("No data retrieved.")
        