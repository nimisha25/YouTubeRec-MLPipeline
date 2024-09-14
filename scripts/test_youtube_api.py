from googleapiclient.discovery import build
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("YOUTUBE_API_KEY")

def main():
    # Your API key here
    api_key = "AIzaSyBBKmDdLrUAetYeZDJ-db6M6T7bvO0lMmE"

    # Build the YouTube API client
    youtube = build("youtube", "v3", developerKey=api_key)

    # Test the API by searching for a query
    # request = youtube.search().list(
    #     part="snippet",
    #     q="comedy",
    #     maxResults=1
    # )
    # response = request.execute()

    # print(response['items'][0]['id']['videoId'])
    # # Print the video title and description
    # for item in response['items']:
    #     print(f"Title: {item['snippet']['title']}")
    #     print(f"Description: {item['snippet']['description']}")
    
    request2 = youtube.videos().list(
        part = "snippet,statistics",
        id = "r8ZZa1sXDSc"
    )
    response2 = request2.execute()
    print(response2)

if __name__ == "__main__":
    main()
