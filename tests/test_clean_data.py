import pytest
import pandas as pd
import tempfile
from scripts.clean_data import clean_watch_history

def test_clean_watch_history():
    """
    Tests if `clean_watch_history` correctly cleans data.
    """
    # Prepare a mock DataFrame
    test_data = {
        "video_id": ["abc123", "def456"],
        "title": ["   Video 1  ", "Video 2"],
        "view_count": [1000, None],
        "like_count": [None, 100],
        "tags": [None, "['anoop soni, crime, show']"],
        "time": ["2023-08-15T12:34:56Z", "2023-08-16T08:15:30Z"],
        "category_id": ["22", None]
    }
    test_df = pd.DataFrame(test_data)

    # Create a temporary file for the mock input
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
        test_input_file = temp_file.name
        test_df.to_csv(test_input_file, index=False)

    # Run the cleaning function
    cleaned_df = clean_watch_history(input_file=test_input_file, return_df=True)

    assert isinstance(cleaned_df, pd.DataFrame)
    assert cleaned_df['view_count'].isnull().sum() == 0  # No nulls in view_count
    assert cleaned_df['like_count'].isnull().sum() == 0  # No nulls in like_count
    assert cleaned_df['tags'].iloc[0] == "[]"  # Tags filled correctly for the first row
    assert cleaned_df['title'].iloc[0] == "Video 1"  # Title stripped of extra spaces
    assert "crime" in cleaned_df['tags'].iloc[1]  # Tags cleaned correctly for the second row
