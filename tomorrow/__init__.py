__version__ = "1.0.0"
__date__ = "05-12-24"
__author__ = "B.Topchiyski"

import os

DATA_PATH = "tomorrow/data/processed/data.csv"
FEATURES = ['AirTemp', 'Press', 'UMR']
TARGETS = ['NO', 'NO2', 'O3', 'RM10']

# Constants and configurations
RAW_DATA_DIR = "tomorrow/data/raw"
OUTPUT_DATA_DIR = "tomorrow/data/processed"
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