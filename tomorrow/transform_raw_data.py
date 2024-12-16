import os

from tomorrow.script.data.read_data import read_day_file, read_hour_file
from tomorrow.script.data.process_data import process_data
from tomorrow.script.data.export_data import export_to_csv

from tomorrow import (
    RAW_DATA_DIR, OUTPUT_DATA_DIR, DAY_FILE, HOUR_FILE, OUTPUT_FILE,
    SHEET_LIST, DAY_FILE_COLS, DAY_FILE_DTYPE, HOUR_FILE_COLS, HOUR_FILE_DTYPE
)

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
    export_columns = ['Date', 'NO', 'NO2', 'AirTemp', 'Press', 'UMR', 'O3', 'RM10']
    export_to_csv(transformed_data, OUTPUT_FILE, export_columns)

    print("Data transformation complete!")

if __name__ == "__main__":
    main()