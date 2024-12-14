import pandas as pd

def read_day_file(filepath, sheet_list, usecols, dtype):
    """
    Reads the daily data file and returns a dictionary of pandas DataFrames.
    """
    try:
        day_data = pd.read_excel(
            filepath,
            sheet_name=sheet_list,
            header=1,
            usecols=usecols,
            dtype=dtype
        )
        print(f"Successfully read daily data from {filepath}")
        return day_data
    except Exception as e:
        print(f"Error reading daily data: {e}")
        raise

def read_hour_file(filepath, sheet_list, usecols, dtype):
    """
    Reads the hourly data file and returns a dictionary of pandas DataFrames.
    """
    try:
        hour_data = pd.read_excel(
            filepath,
            sheet_name=sheet_list,
            header=1,
            usecols=usecols,
            dtype=dtype
        )
        print(f"Successfully read hourly data from {filepath}")
        return hour_data
    except Exception as e:
        print(f"Error reading hourly data: {e}")
        raise