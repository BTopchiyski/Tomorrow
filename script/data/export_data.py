def export_to_csv(dataframe, output_filepath, columns):
    """
    Exports the given DataFrame to a CSV file.
    """
    try:
        dataframe.to_csv(
            output_filepath,
            index=True,
            columns=columns
        )
        print(f"Data successfully exported to {output_filepath}")
    except Exception as e:
        print(f"Error exporting data: {e}")
        raise