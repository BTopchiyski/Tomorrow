import pandas as pd

def process_data(day_data, hour_data):
    """
    Processes and merges daily and hourly data into a single DataFrame.
    """
    merged_df = pd.DataFrame()

    try:
        for sheet_name, day_sheet in day_data.items():
            for i in range(day_sheet['Date'].size):
                # Extract daily data
                date = day_sheet.iloc[i]['Date']
                o3 = day_sheet.iloc[i]['O3']
                rm10 = day_sheet.iloc[i]['RM10']

                # Extract matching hourly data
                hour_sheet = hour_data[sheet_name]
                filtered_hour_data = hour_sheet[hour_sheet['Date'].str.contains(date)]

                # Add daily data columns to the hourly data
                filtered_hour_data.insert(2, 'O3', o3)
                filtered_hour_data.insert(3, 'RM10', rm10)

                # Append to the merged DataFrame
                merged_df = merged_df._append(filtered_hour_data, ignore_index=True)

        # Drop rows with missing data
        merged_df = merged_df.dropna()

        # Round numerical columns
        rounded_df = merged_df.round({
            'NO': 2, 
            'NO2': 2, 
            'O3': 2, 
            'RM10': 2, 
            'AirTemp': 1, 
            'UMR': 1, 
            'Press': 0
        })
        
        print("Data processed successfully.")
        return rounded_df

    except Exception as e:
        print(f"Error processing data: {e}")
        raise