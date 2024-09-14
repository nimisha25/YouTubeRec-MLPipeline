import boto3
import pandas as pd

def loading_to_s3(local_file_path, bucket_name, s3_file_path):
    """
    Uploads a local CSV file to an Amazon S3 bucket.

    This function reads a cleaned CSV file, initializes an S3 client using the `boto3` library, 
    and uploads the file to the specified Amazon S3 bucket.

    Parameters:
    -----------
    local_file_path : str
        The local file path to the cleaned CSV file that needs to be uploaded to S3.
    bucket_name : str
        The name of the target S3 bucket where the file will be uploaded.
    s3_file_path : str
        The desired file path (key) within the S3 bucket where the file will be stored.

    Returns:
    --------
    None:
        The function uploads the file to S3 and does not return any value.
    """
    
    cleaned_watch_history = pd.read_csv(local_file_path)
    s3 = boto3.client('s3')
    s3.upload_file(Filename=local_file_path, Bucket=bucket_name, Key=s3_file_path)
