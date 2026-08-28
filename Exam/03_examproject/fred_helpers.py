""" helper functions for Problem 1: downloading state-level data from FRED

Used in Opgave1.ipynb to keep the notebook focused on the analysis instead
of the mechanics of calling the FRED API.

"""

import pandas as pd
import requests

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def download_fred_data(series_id, api_key):
    """Download one FRED series and return it as a Series indexed by year."""
    parameters = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json"
    }
    response = requests.get(FRED_URL, params=parameters, timeout=30)

    if response.status_code != 200:
        print(f"Error downloading {series_id}:", response.status_code)
        print(response.text)
        return pd.Series(dtype="float64", name=series_id)

    observations = response.json()["observations"]
    data = pd.DataFrame(observations)
    data["year"] = pd.to_datetime(data["date"]).dt.year
    data["value"] = pd.to_numeric(data["value"], errors="coerce")

    result = data.set_index("year")["value"]
    result.name = series_id

    return result


def download_all_states(suffix, states, api_key):
    """Download one FRED series for every state and collect them in a data frame."""
    downloaded_series = {}

    for number, state in enumerate(states, start=1):
        series_id = f"{state}{suffix}"
        print(f"{number}/{len(states)}: downloading {series_id}")
        downloaded_series[state] = download_fred_data(series_id, api_key)

    result = pd.DataFrame(downloaded_series)

    return result.sort_index()
