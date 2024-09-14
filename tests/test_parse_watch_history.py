import pandas as pd
from scripts.parse_watch_history import parse_watch_history
from unittest.mock import patch, mock_open

def test_parse_watch_history():
    """
    Tests if `parse_watch_history` parses JSON data correctly.
    """
    # Use the mock JSON file for testing
    file_name = "tests/test_watch_history.json"
    
    # Run the function with the return_df flag set to True
    result = parse_watch_history(file_name, return_df=True)
    
    # Assertions
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert set(result.columns) == {"title", "time", "video_id", "channel_name"}
    assert len(result) == 1  # Only one entry should be after the cutoff date
    assert result['video_id'].iloc[0] == 'abc123'
