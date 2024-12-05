import os
from script.data.read_data import read_day_file, read_hour_file
from script.data.process_data import process_data
from script.data.export_data import export_to_csv

# Constants and configurations
RAW_DATA_DIR = "./data/raw"
OUTPUT_DATA_DIR = "./data/processed"
DAY_FILE = os.path.join(RAW_DATA_DIR, "day.xls")
HOUR_FILE = os.path.join(RAW_DATA_DIR, "hour.xls")
OUTPUT_FILE = os.path.join(OUTPUT_DATA_DIR, "data.csv")
SHEET_LIST = ['pavlovo', 'drujba', 'hipodruma']

# Day file configuration
DAY_FILE_COLS = ['Date', 'O3', 'RM10']
DAY_FILE_DTYPE = {'Date': str, 'O3': float, 'RM10': float}

# Hour file configuration
HOUR_FILE_COLS = ['Date', 'NO', 'NO2', 'AirTemp', 'Press', 'UMR']
HOUR_FILE_DTYPE = {'Date': str, 'NO': float, 'NO2': float, 'AirTemp': float, 'Press': float, 'UMR': float}

def main():
    # Step 1: Read raw data
    print("Reading data...")
    day_data = read_day_file(DAY_FILE, SHEET_LIST, DAY_FILE_COLS, DAY_FILE_DTYPE)
    hour_data = read_hour_file(HOUR_FILE, SHEET_LIST, HOUR_FILE_COLS, HOUR_FILE_DTYPE)

    # Step 2: Process the data
    print("Processing data...")
    transformed_data = process_data(day_data, hour_data)

    # Step 3: Export the transformed data
    print("Exporting data...")
    export_columns = ['NO', 'NO2', 'AirTemp', 'Press', 'UMR', 'O3', 'RM10']
    export_to_csv(transformed_data, OUTPUT_FILE, export_columns)

    print("Data transformation complete!")

if __name__ == "__main__":
    main()
