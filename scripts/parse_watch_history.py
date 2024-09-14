import json
import pandas as pd
def parse_watch_history(file_name, return_df=False) :
    """
    Parses and filters watch history data from a Google Takeout JSON file.

    This function processes the watch history data by extracting relevant fields (e.g., title, time, video ID, channel name) 
    and filtering out entries that are ads or lack required information (e.g., title URL or subtitles). It converts the 
    time data to a standard datetime format and filters entries to include only those after August 1, 2023.

    Parameters:
    -----------
    file_name : str
        The file path of the JSON file containing the watch history data.
    return_df : bool, optional
        If True, the function will return the filtered DataFrame. If False, it will save the filtered DataFrame 
        as a CSV file to 'data/filtered_watch_history.csv'. Default is False.

    Returns:
    --------
    pd.DataFrame or None:
        If `return_df` is True, returns the filtered DataFrame with columns:
        - 'title': The title of the video.
        - 'time': The time the video was watched (in datetime format).
        - 'video_id': The YouTube video ID.
        - 'channel_name': The name of the video channel.

        If `return_df` is False, the DataFrame is saved as a CSV file, and the function returns None.
    """
    with open(file_name, 'r') as file:
        data = json.load(file)

    parsed_data = []
    for entry in data :
        title = entry.get('title')
        time = entry.get('time')
        title_url = entry.get('titleUrl')
        details = entry.get('details')
        subtitles = entry.get('subtitles')

        if details is not None :
            name = details[0].get('name')
            if name == 'From Google Ads' :
                continue
        if not subtitles or not title_url:
            continue
        
        video_id = title_url.split('v=')[-1]
        parsed_data.append({
        'title': title,
        'time': time,
        'video_id': video_id,
        'channel_name': subtitles[0].get('name') if subtitles else None
        })
    watch_history = pd.DataFrame(parsed_data, columns = ['title', 'time' , 'video_id', 'channel_name',])
    #print(watch_history)

    watch_history['time'] = pd.to_datetime(watch_history['time'], format='ISO8601', errors='coerce')
    cutoff_date = pd.Timestamp('2023-08-01').tz_localize('UTC')

    filtered_watch_history = watch_history[watch_history['time'] >= cutoff_date]
    if not return_df :
        filtered_watch_history.to_csv('data/filtered_watch_history.csv', index=False)

    if return_df:
        return filtered_watch_history

if __name__ == "__main__":
    file_name = '/Users/nimishamalik/Desktop/ETLproject/data/watch-history.json'  # Update with your file path
    parse_watch_history(file_name)

