import pandas as pd
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
import numpy as np
import datetime
import random
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()

# Access the API key
api_key = os.getenv("YOUTUBE_API_KEY")

def get_time_of_day():
    """
    Determines the current time of day based on the system's local time.

    Returns:
    --------
    str:
        The time of day as 'morning', 'afternoon', 'evening', or 'night' based on the current hour.
    """
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        return 'morning'
    elif 12 <= current_hour < 17:
        return 'afternoon'
    elif 17 <= current_hour < 21:
        return 'evening'
    else:
        return 'night'

def fetch_new_videos(cluster_label, max_results=10):
    """
    Fetches new videos from YouTube API based on the given cluster label.

    This function uses the YouTube Data API to search for videos matching the cluster label, 
    ordered by view count. It returns a list of video IDs and titles.

    Parameters:
    -----------
    cluster_label : str
        The cluster label used as the search query.
    max_results : int, optional
        The maximum number of videos to fetch (default is 10).

    Returns:
    --------
    list of tuples:
        A list of tuples where each tuple contains a video ID and its corresponding title.
    """
    youtube = build('youtube', 'v3', developerKey=api_key)

    random_seed = random.randint(1,100)

    search_response = youtube.search().list(
        q=f"{cluster_label} {random_seed}", 
        part='snippet',
        type='video',
        maxResults=max_results,
        order='viewCount', 
        videoEmbeddable='true', 
    ).execute()

    video_ids = []
    for item in search_response['items']:
        video_id = item['id']['videoId']
        video_title = item['snippet']['title']
        video_ids.append((video_id, video_title))

    return video_ids

def get_video_details(video_ids):
    """
    Fetches video details such as view count and like count using the YouTube Data API.

    This function takes a list of video IDs, retrieves their statistics (view count, like count), 
    and returns these details.

    Parameters:
    -----------
    video_ids : list of tuples
        A list of tuples where each tuple contains a video ID and its corresponding title.

    Returns:
    --------
    list of dict:
        A list of dictionaries containing video ID, title, view count, and like count.
    """
    youtube = build('youtube', 'v3', developerKey='AIzaSyBBKmDdLrUAetYeZDJ-db6M6T7bvO0lMmE')
    
    video_details = []
    for video_id, video_title in video_ids:
        details = youtube.videos().list(
            part='statistics',
            id=video_id
        ).execute()

        for video in details['items']:
            view_count = int(video['statistics'].get('viewCount', 0))
            like_count = int(video['statistics'].get('likeCount', 0))
            video_details.append({
                'video_id': video_id,
                'title': video_title,
                'view_count': view_count,
                'like_count': like_count
            })

    return video_details

def filter_videos_by_engagement(videos, cluster_engagement_metrics, max_results = 5):
    """
    Filters videos based on engagement metrics (view and like counts) of the cluster.

    This function compares the videos' engagement metrics (view and like counts) against the cluster's 
    mean and standard deviation. If no videos fall within one standard deviation, it widens the threshold 
    to two standard deviations, and as a last resort, returns the top N videos by view count.

    Parameters:
    -----------
    videos : list of dict
        A list of video details (view count, like count) to be filtered.
    cluster_engagement_metrics : pd.DataFrame
        Engagement metrics (mean and standard deviation of views and likes) for the cluster.
    max_results : int, optional
        The maximum number of filtered videos to return (default is 5).

    Returns:
    --------
    list of dict:
        A filtered list of videos that match the engagement trends of the cluster.
    """
    filtered_videos = []
    for video in videos:
        view_count = video['view_count']
        like_count = video['like_count']
        
        mean_view = cluster_engagement_metrics[('view_count','mean')]
        std_view = cluster_engagement_metrics[('view_count','std')]
        mean_like = cluster_engagement_metrics[('like_count','mean')]
        std_like = cluster_engagement_metrics[('like_count','std')]

        if (mean_view - std_view <= view_count <= mean_view + std_view) and \
           (mean_like - std_like <= like_count <= mean_like + std_like):
            filtered_videos.append(video)

        if not filtered_videos:
            for video in videos:
                view_count = video['view_count']
                like_count = video['like_count']
                
                if (mean_view - 2 * std_view <= view_count <= mean_view + 2 * std_view) and \
                (mean_like - 2 * std_like <= like_count <= mean_like + 2 * std_like):
                    filtered_videos.append(video)
    
        if not filtered_videos:
            filtered_videos = sorted(videos, key=lambda x: x['view_count'], reverse=True)[:max_results]

    return filtered_videos

def kmeans_elbow(df) :
    """
    Plots the elbow method for determining the optimal number of clusters in K-Means.

    This function calculates the within-cluster sum of squares (WCSS) for different numbers 
    of clusters (1 to 10) and plots the results to help determine the optimal number of clusters.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to be clustered.

    Returns:
    --------
    None
        The function plots the elbow curve.
    """
    wcss = []
    for k in range (1, 11) :
        kmeans=KMeans(n_clusters=k,init='k-means++', random_state=42)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1,11), wcss)
    # plt.show()

def pca(df) :
    """
    Performs PCA (Principal Component Analysis) on the given DataFrame and applies K-Means clustering.

    This function reduces the dimensionality of the data to 2 components using PCA, 
    applies K-Means clustering, and visualizes the clusters.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to be reduced and clustered.

    Returns:
    --------
    np.ndarray:
        The cluster labels for each data point.
    """
    pca = PCA(n_components = 2)
    df_pca = pca.fit_transform(df)
    # plt.plot(range(1,608), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '-')
    # kmeans_pca2 = KMeans(n_clusters = 2, init = 'k-means++', random_state=42)
    # kmeans_pca2.fit(df_pca)
    # kmeans_pca4 = KMeans(n_clusters = 4, init = 'k-means++', random_state=42)
    # kmeans_pca4.fit(df_pca)
    kmeans_pca3 = KMeans(n_clusters = 3, init = 'k-means++', random_state=42)
    kmeans_pca3.fit(df_pca)
    # kmeans_pca5 = KMeans(n_clusters = 5, init = 'k-means++', random_state=42)
    # kmeans_pca5.fit(df_pca)
    
    # print("for 2:" ,silhouette_score(df_pca, kmeans_pca2.fit_predict(df_pca)))
    # print("for 4:" ,silhouette_score(df_pca, kmeans_pca4.fit_predict(df_pca)))
    # print("for 3:" ,silhouette_score(df_pca, kmeans_pca3.fit_predict(df_pca)))
    # print("for 5:" ,silhouette_score(df_pca, kmeans_pca3.fit_predict(df_pca)))
    clusters = kmeans_pca3.labels_

    plt.figure(figsize=(10, 6))
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='viridis', marker='o', s=50)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Clusters visualized on PCA-reduced data')
    plt.colorbar(label='Cluster Label')
    # plt.show()
    return clusters


def analyze_clusters(df, clusters):
    """
    Analyzes clusters by computing descriptive statistics for each cluster.

    This function groups the data by cluster and computes statistics (mean, median, std) for various 
    features such as view count, like count, and time-related features.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data.
    clusters : np.ndarray
        The cluster labels for each data point.

    Returns:
    --------
    pd.DataFrame:
        A DataFrame containing descriptive statistics for each cluster.
    """

    df['cluster'] = clusters
    
    non_embedding_cols = [col for col in df.columns if not col.startswith(('tag_embedding', 'title_avg_embedding'))]
    
    cluster_analysis = df[non_embedding_cols].groupby('cluster').agg({
        'view_count': ['mean', 'median', 'std'],
        'like_count': ['mean', 'median', 'std'],
        'time_hour': ['mean', 'median', 'std'],
        'day_category_morning': 'mean', 
        'day_category_afternoon': 'mean',
        'day_category_evening': 'mean',
        'day_category_night': 'mean',
    })
    
    return cluster_analysis

def find_representative_videos(df, clusters, num_videos=3):
    """
    Selects representative videos from each cluster.

    This function selects a random sample of representative videos from each cluster.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing video data (e.g., titles, tags).
    clusters : np.ndarray
        The cluster labels for each data point.
    num_videos : int, optional
        The number of representative videos to select per cluster (default is 3).

    Returns:
    --------
    dict:
        A dictionary where each key is a cluster ID and the value is a DataFrame 
        with selected representative videos from that cluster.
    """  
    representative_videos = {}
    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        selected_indices = np.random.choice(cluster_indices, size=num_videos, replace=False)
        representative_videos[cluster_id] = df.iloc[selected_indices][['title', 'tags']] 
    return representative_videos

def label_clusters(x) :
    """
    This function maps cluster IDs to human-readable labels such as 'Crime and Comedy', 
    'Shorts and Entertainment', and 'Education and Technology'.

    Parameters:
    -----------
    x : int
        The cluster ID.

    Returns:
    --------
    str:
        The human-readable label for the cluster.
    """
    if x == 0 :
        return 'Crime and Comedy'
    elif x == 1 :
        return 'Shorts and Entertainment'
    else :
        return 'Education and Technology'
    # return df 

def time_of_day_analysis(df):
    """
    This function calculates the distribution of viewing times (morning, afternoon, evening, night) 
    for each cluster and identifies the dominant clusters by time of day.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing day categories and cluster labels.

    Returns:
    --------
    tuple:
        - day_category_distribution: The distribution of day categories for each cluster.
        - dominant_clusters: The dominant cluster for each time of day.
    """
    df['day_category'] = df[['day_category_morning', 'day_category_afternoon', 'day_category_evening', 'day_category_night']].idxmax(axis=1)
    
    df['day_category'] = df['day_category'].str.replace('day_category_', '')
    day_category_distribution = df.groupby(['cluster_label', 'day_category']).size()
    day_category_distribution = day_category_distribution.groupby(level=0).apply(lambda x: x / x.sum())
    dominant_clusters = day_category_distribution.groupby('day_category').idxmax().apply(lambda x: x[0])
    
    return day_category_distribution, dominant_clusters

def analyze_engagement(df, cluster_labels_column='cluster_label'):
    """
    Analyzes the engagement metrics for each cluster based on view and like counts.

    This function groups the data by clusters and calculates engagement statistics (mean, median, std) 
    for views and likes. It also identifies the cluster with the highest average views and likes.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the engagement data.
    cluster_labels_column : str, optional
        The column name representing cluster labels (default is 'cluster_label').

    Returns:
    --------
    tuple:
        - engagement_metrics: A DataFrame containing engagement statistics for each cluster.
        - highest_avg_views: The cluster with the highest average views.
        - highest_avg_likes: The cluster with the highest average likes.
    """

    engagement_metrics = df.groupby(cluster_labels_column).agg({
        'view_count': ['mean', 'median', 'std'],
        'like_count': ['mean', 'median', 'std']
    })

    highest_avg_views = engagement_metrics['view_count']['mean'].idxmax()
    highest_avg_likes = engagement_metrics['like_count']['mean'].idxmax()

    return engagement_metrics, highest_avg_views, highest_avg_likes


if __name__ == '__main__' :
    
    # Step 1: Load and preprocess the dataset
    df = pd.read_csv("data/final_feature_set_bert.csv")
    # df = rename_embedding_columns(df)
    
    # Step 2: Perform clustering
    clusters = pca(df)
    df['cluster'] = clusters
    df['cluster_label'] = df['cluster'].apply(label_clusters)   # Manual renaming functions

    # Step 3: Save the clustered dataset
    df.to_csv("data/clustered_dataset.csv", index=False)
    
    # Step 4: Perform time of day analysis
    day_category_distribution, dominant_clusters = time_of_day_analysis(df)
    print('day category distribution', day_category_distribution)
    print('dominant cluster', dominant_clusters)
    
    # Step 5: Perform engagement metrics analysis
    engagement_metrics, highest_avg_views, highest_avg_likes = analyze_engagement(df)
    print(engagement_metrics)
    
    # Step 6: Detect the current time of day
    time_of_day = get_time_of_day()
    print(f"Current time of day: {time_of_day}")
    
    # Step 7: Fetch new videos based on the dominant cluster for the time of day
    new_videos = fetch_new_videos(dominant_clusters[time_of_day])
    
    # Step 8: Get details (view count, like count) for the fetched videos
    video_details = get_video_details(new_videos)
    
    # Step 9: Filter fetched videos based on engagement metrics of the cluster
    cluster_engagement = engagement_metrics.loc[dominant_clusters[time_of_day]]
    filtered_videos = filter_videos_by_engagement(video_details, cluster_engagement)
    
    # Step 10: Print the filtered recommended videos
    print("\nRecommended Videos Based on Engagement Trends:")
    for video in filtered_videos:
        print(f"Title: {video['title']}, Views: {video['view_count']}, Likes: {video['like_count']}")