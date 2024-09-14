from googleapiclient.discovery import build
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("YOUTUBE_API_KEY")
youtube = build('youtube', 'v3', developerKey=api_key)

def fetch_video_details(video_id) :
    """
    Fetches video details for a given list of video IDs from the YouTube Data API.

    This function retrieves metadata for each video, including view count, like count, category ID, and tags. 
    It sends a request to the YouTube API using the provided video IDs, batches the results if necessary, 
    and returns the details in a structured format.

    Parameters:
    -----------
    video_id : list of str
        A list of YouTube video IDs for which to fetch metadata.

    Returns:
    --------
    list of dict:
        A list of dictionaries, each containing metadata for a video. The dictionary includes:
        - 'video_id': The ID of the video.
        - 'view_count': The number of views on the video.
        - 'like_count': The number of likes on the video.
        - 'category_id': The category ID of the video.
        - 'tags': The tags associated with the video.

        Returns an empty list if the video ID is invalid or if there is an error in fetching the details.
    """
    if not video_id :
        return []
    try :

        request = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(video_id)  
        )
        response = request.execute()

        video_data = []
        for item in response['items']:
            video_data.append({
                'video_id': item['id'],
                'view_count': item['statistics'].get('viewCount'),
                'like_count': item['statistics'].get('likeCount'),
                'category_id': item['snippet'].get('categoryId'),
                'tags': item['snippet'].get('tags')
            })
        
        return video_data
    except Exception as e :
        print("Error fetching data for video IDs")
        return []


def enrich_watch_history(input_file, return_df = False) :
    """
    Enriches the watch history data with additional metadata from the YouTube Data API.

    This function reads the watch history data from the specified CSV file, fetches additional metadata 
    for each video (e.g., view count, like count, category ID, tags), and merges it with the existing data. 
    The enriched dataset is either returned as a DataFrame or saved to a CSV file.

    Parameters:
    -----------
    input_file : str
        The file path of the CSV file containing the watch history data.
    return_df : bool, optional
        If True, the function will return the enriched DataFrame. If False, it will save the enriched DataFrame 
        as a CSV file to 'data/enriched_watch_history.csv'. Default is False.

    Returns:
    --------
    pd.DataFrame or None:
        If `return_df` is True, returns the enriched DataFrame. Otherwise, saves the DataFrame and returns None.
    """

    filtered_watch_history = pd.read_csv(input_file)
    all_video_metadata = []
    video_ids = filtered_watch_history['video_id'].tolist()

    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        metadata = fetch_video_details(batch)
        all_video_metadata.extend(metadata)


    metadata_df = pd.DataFrame(all_video_metadata)
    metadata_df = metadata_df.drop_duplicates(subset = ['video_id'])
    enriched_watch_history = pd.merge(filtered_watch_history, metadata_df, on='video_id', how='left')

    if return_df :
        return enriched_watch_history
    else :
        enriched_watch_history.to_csv('data/enriched_watch_history.csv', index=False)

