import pandas as pd

def clean_watch_history(input_file, return_df = False) :
    """
    Cleans and processes the watch history data from a CSV file.

    This function performs several cleaning steps on the input watch history data:
    1. Fills missing values in `view_count` and `like_count` with 0.
    2. Fills missing or empty values in the `tags` column with '[]'.
    3. Standardizes the format of the `time` column to a datetime format.
    4. Strips whitespace from text in the `title` column.
    5. Cleans the `tags` column by removing invalid characters (anything outside of alphanumeric, brackets, and commas).
    6. Converts the `category_id` column to numeric, filling any non-numeric or missing values with -1.

    Parameters:
    -----------
    input_file : str
        The file path of the CSV file containing enriched watch history data.
    return_df : bool, optional
        If True, the function will return the cleaned DataFrame. If False, it will save the cleaned DataFrame as a CSV
        file to 'data/cleaned_watch_history.csv'. Default is False.

    Returns:
    --------
    pd.DataFrame or None:
        If `return_df` is True, returns the cleaned DataFrame. Otherwise, saves the DataFrame and returns None.
    """
    
    enriched_watch_history = pd.read_csv(input_file)

    enriched_watch_history['view_count'].fillna(0, inplace = True)
    enriched_watch_history['like_count'].fillna(0, inplace=True)
    enriched_watch_history['tags'].fillna('[]', inplace=True)

    enriched_watch_history['tags'] = enriched_watch_history['tags'].apply(
    lambda x: "[]" if pd.isna(x) or x == "" or x.strip() == "[]" else x)

    enriched_watch_history['time'] = pd.to_datetime(enriched_watch_history['time'], errors='coerce')

    enriched_watch_history['title'] = enriched_watch_history['title'].str.strip()
    enriched_watch_history['tags'] = enriched_watch_history['tags'].str.replace(r"[^\[\]a-zA-Z0-9, ]", "", regex=True)

    enriched_watch_history['category_id'] = pd.to_numeric(enriched_watch_history['category_id'], errors='coerce')
    enriched_watch_history['category_id'].fillna(-1, inplace=True)  # Use -1 for unknown categories

    if not return_df :
        enriched_watch_history.to_csv('data/cleaned_watch_history.csv', index=False)
    else :
        return enriched_watch_history