from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import pandas as pd
# Append the path to the 'scripts' and 'ml_task' directories
sys.path.append('/Users/nimishamalik/Desktop/ETLproject/scripts')
sys.path.append('/Users/nimishamalik/Desktop/ETLproject/ml_task')

from parse_watch_history import parse_watch_history
from fetch_youtube_metadata import enrich_watch_history
from clean_data import clean_watch_history
from upload_to_s3 import loading_to_s3
# from clustering_task import 
from retreive_data import get_file_from_s3
from preprocess_features_bert import preprocess_data
from clustering_task import pca, time_of_day_analysis, fetch_new_videos, get_video_details, filter_videos_by_engagement, label_clusters, analyze_engagement, get_time_of_day



# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 9, 7),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Define the DAG
with DAG(
    'youtube_etl_pipeline',
    default_args=default_args,
    description='ETL pipeline for YouTube project',
    schedule_interval=None,
) as dag:

    # Define the tasks
    extract_task = PythonOperator(
        task_id='extract_data',
        python_callable=parse_watch_history,  # This will call the function that parses the watch history
        op_args=['/Users/nimishamalik/Desktop/ETLproject/data/watch-history.json'],
    )

    transform_task = PythonOperator(
        task_id='transform_data',
        python_callable=enrich_watch_history,  # This will call the function that fetches YouTube metadata and enriches the data
        op_kwargs={'input_file': '/Users/nimishamalik/Desktop/ETLproject/data/filtered_watch_history.csv', 'return_df': False},
    )

    clean_task = PythonOperator(
        task_id='clean_data',
        python_callable=clean_watch_history,  # This will call the function that cleans and standardizes the data
        op_kwargs={'input_file': '/Users/nimishamalik/Desktop/ETLproject/data/enriched_watch_history.csv', 'return_df': False},
    )

    load_task = PythonOperator(
        task_id='load_data',
        python_callable=loading_to_s3,  # This will call the function that uploads the final data to S3
        op_kwargs={
            'local_file_path': '/Users/nimishamalik/Desktop/ETLproject/data/cleaned_watch_history.csv',
            'bucket_name': 'my-youtube-project-bucket',
            's3_file_path': 'output/cleaned_watch_history.csv'
        },  # Update with parameters
    )

    retreive_task = PythonOperator(
        task_id='retreive_task',
        python_callable=get_file_from_s3,
        op_kwargs={
            'bucket_name' : "my-youtube-project-bucket",
            's3_file_path' : "project-data/cleaned_watch_history.csv",
            'local_file_path' : "data/cleaned_watch_history.csv",
        }
    )

    preprocess_task = PythonOperator(
        task_id='preprocess_task',
        python_callable=preprocess_data,
        op_kwargs={
            'filename' : "data/cleaned_watch_history.csv"
        }
    )

    def clustering_logic(filename):
        df = pd.read_csv(filename)
        # df = rename_embedding_columns(df)
        clusters = pca(df)
        df['cluster'] = clusters
        df['cluster_label'] = df['cluster'].apply(label_clusters)

        df.to_csv("data/clustered_dataset.csv", index=False)

        day_category_distribution, dominant_clusters = time_of_day_analysis(df)
        engagement_metrics, highest_avg_views, highest_avg_likes = analyze_engagement(df)

        time_of_day = get_time_of_day()
        new_videos = fetch_new_videos(dominant_clusters[time_of_day])
        video_details = get_video_details(new_videos)
        cluster_engagement = engagement_metrics.loc[dominant_clusters[time_of_day]]
        filtered_videos = filter_videos_by_engagement(video_details, cluster_engagement)

        print("\nRecommended Videos Based on Time and Engagement Trends:")
        for video in filtered_videos:
            print(f"Title: {video['title']}, Views: {video['view_count']}, Likes: {video['like_count']}")

    clustering_task = PythonOperator(
        task_id='clustering_task',
        python_callable=clustering_logic,
        op_kwargs={'filename': "data/final_feature_set_bert.csv"},
    )


    # Set task dependencies
    extract_task >> transform_task >> clean_task >> load_task >> retreive_task >> preprocess_task >> clustering_task
