import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from scripts.fetch_youtube_metadata import enrich_watch_history
import ast

@patch('scripts.fetch_youtube_metadata.youtube.videos')
@patch('pandas.read_csv')
def test_enrich_watch_history(mock_read_csv, mock_videos):
    """
    Tests if `enrich_watch_history` enriches watch history with YouTube metadata.
    """
    # Prepare the mock response for the YouTube API
    mock_videos().list().execute.return_value = {
        "items": [
            {
                "id": "abc123",
                "statistics": {"viewCount": "1000", "likeCount": "50"},
                "snippet": {"categoryId": "22", "tags": ["tag1", "tag2"]}
            },
            {
                "id": "def456",
                "statistics": {"viewCount": "1500", "likeCount": "100"},
                "snippet": {"categoryId": "22", "tags": ["tag3", "tag4"]}
            }
        ]
    }

    # Mock the filtered watch history CSV content
    test_data = {
        "video_id": ["abc123", "def456"],
        "title": ["Video 1", "Video 2"],
        "time": ["2023-08-15T12:34:56Z", "2023-08-16T08:15:30Z"],
        "channel_name": ["Channel 1", "Channel 2"]
    }
    test_df = pd.DataFrame(test_data)
    mock_read_csv.return_value = test_df  # Mock the read_csv to return the DataFrame

    enriched_df = enrich_watch_history('test_input.csv', return_df=True)

    assert isinstance(enriched_df, pd.DataFrame)
    assert not enriched_df.empty
    assert 'view_count' in enriched_df.columns
    assert enriched_df.loc[enriched_df['video_id'] == 'abc123', 'view_count'].iloc[0] == "1000"
    assert enriched_df.loc[enriched_df['video_id'] == 'def456', 'view_count'].iloc[0] == "1500"

    # Ensure the merged data has the correct size and no duplicates
    assert enriched_df.shape[0] == 2
    assert enriched_df.duplicated(subset='video_id').sum() == 0

    # Validate the tags field by checking if it's a list or needs conversion
    tags_value = enriched_df.loc[enriched_df['video_id'] == 'abc123', 'tags'].iloc[0]
    if isinstance(tags_value, str):
        tags_value = ast.literal_eval(tags_value) 
    assert tags_value == ["tag1", "tag2"]

if __name__ == "__main__":
    pytest.main()
