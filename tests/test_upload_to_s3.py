import pytest
from unittest.mock import patch, MagicMock
from scripts.upload_to_s3 import loading_to_s3

@patch('scripts.upload_to_s3.boto3.client')
def test_loading_to_s3(mock_s3_client):
    """
    Tests if `loading_to_s3` uploads a file to S3.
    """
    # Mock the S3 client and its behavior
    mock_s3 = MagicMock()
    mock_s3_client.return_value = mock_s3

    # Run the function with mocked parameters
    local_file = 'data/cleaned_watch_history.csv'
    bucket = 'my-bucket'
    s3_path = 'output/cleaned_watch_history.csv'

    loading_to_s3(local_file, bucket, s3_path)

    mock_s3.upload_file.assert_called_once_with(
        Filename=local_file,
        Bucket=bucket,
        Key=s3_path
    )

if __name__ == "__main__":
    pytest.main()
