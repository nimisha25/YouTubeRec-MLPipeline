import boto3
import pandas as pd


def get_file_from_s3(bucket_name, s3_file_path, local_file_path) :
    """
    Downloads a file from an Amazon S3 bucket and loads it as a pandas DataFrame.

    This function downloads a CSV file from the specified S3 bucket and file path, saves it locally, 
    and reads the file into a pandas DataFrame for further use.

    Parameters:
    -----------
    bucket_name : str
        The name of the S3 bucket from which the file is to be downloaded.
    s3_file_path : str
        The file path (key) within the S3 bucket of the file to be downloaded.
    local_file_path : str
        The local file path where the downloaded file will be saved.

    Returns:
    --------
    pd.DataFrame:
        A pandas DataFrame containing the contents of the downloaded CSV file.
    """
    session = boto3.Session()
    s3 = session.client("s3")

    s3.download_file(Bucket=bucket_name, Key=s3_file_path, Filename=local_file_path)
    df = pd.read_csv(local_file_path)
    return df

if __name__ == "__main__" :
    bucket_name = "my-youtube-project-bucket"
    s3_file_path = "project-data/cleaned_watch_history.csv"
    local_file_path = "data/cleaned_watch_history.csv"
    
    # Retrieve the data
    df = get_file_from_s3(bucket_name, s3_file_path, local_file_path)
    print(df.head())  # Verify that the data is loaded correctly

