from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_bert_embeddings(text):
    """
    Generates BERT embeddings for a given text (titles and tags)

    Parameters:
    -----------
    text : str
        The input text for which embeddings are to be generated.

    Returns:
    --------
    np.ndarray:
        A NumPy array representing the average BERT embeddings for the input text.
    """

    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**tokens)
  
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.numpy()

def process_titles_and_tags(df):
    """
    This function applies the `get_bert_embeddings` function to the 'title' and 'tags' columns of the DataFrame, generating embeddings for both titles and tags. If the 'tags' column is empty or missing, it returns a zero embedding.

    Parameters:
    -----------
    df : pd.DataFrame
        A DataFrame containing 'title' and 'tags' columns.

    Returns:
    --------
    pd.DataFrame:
        The DataFrame with additional columns for title and tag embeddings.
    """
    df['title_avg_embedding'] = df['title'].apply(lambda x: get_bert_embeddings(x))
    df['tag_embedding'] = df['tags'].apply(lambda x: get_bert_embeddings(x) if x else np.zeros(model.config.hidden_size))
    return df

def preprocess_data(filename) :
    """
    Preprocesses the input CSV file by processing titles, tags, and other features.

    This function reads a CSV file, processes the titles and tags using BERT embeddings, processes time-related columns, applies one-hot encoding to the day category, and combines all features into a final standardized feature set. The final feature set is saved to a CSV file.

    Parameters:
    -----------
    filename : str
        The file path of the input CSV file.

    Returns:
    --------
    pd.DataFrame:
        A processed DataFrame with combined features ready for modeling.
    """
    df = pd.read_csv(filename)

    # Process titles and tags using BERT
    df['title'] = df['title'].apply(lambda x: x.lower())
    df['tags'] = df['tags'].apply(lambda x: [] if pd.isna(x) else x.lower() if isinstance(x, str) else x)
    df = process_titles_and_tags(df)

    # Processing 'time'
    df = df.dropna(subset=['time'])
    df['time'] = pd.to_datetime(df['time'])
    df['local_time'] = df['time'].dt.tz_convert('US/Eastern')
    df = get_time_details(df)

    # One hot encoding the day category
    df['day_category'] = df['time_hour'].apply(ret_day_category)
    df = pd.get_dummies(df, columns=['day_category'])

    # Combine features into a final feature set
    df_processed = combine_features(df)
    df_processed.columns = df_processed.columns.astype(str)

    # Standardize the features
    scaler = StandardScaler()
    df_processed.iloc[:, :] = scaler.fit_transform(df_processed)

    df_processed.to_csv("data/final_feature_set_bert.csv", index=False)

    return df_processed

def combine_features(df) :
    """
    Combines various features into a final feature set.

    This function concatenates BERT embeddings for titles and tags, along with numeric, time, and one-hot encoded features into a final DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing embeddings, numeric, and time features.

    Returns:
    --------
    pd.DataFrame:
        A DataFrame containing the combined feature set.
    """
    title_emb_df = pd.DataFrame(df['title_avg_embedding'].tolist(), index=df.index)
    tag_emb_df = pd.DataFrame(df['tag_embedding'].tolist(), index=df.index)
    title_emb_df.columns = [f'title_{i}' for i in range(title_emb_df.shape[1])]
    tag_emb_df.columns = [f'tag_{i}' for i in range(tag_emb_df.shape[1])]

    numeric_features = df[['view_count', 'like_count']]
    time_features = df[['time_hour']]
    one_hot_features = df[['day_category_morning', 'day_category_afternoon', 'day_category_evening', 'day_category_night']]
    final_feature_set = pd.concat([title_emb_df, tag_emb_df, numeric_features, time_features, one_hot_features], axis=1)  
    return final_feature_set
    

def ret_day_category(hr) :
    """
    Categorizes the time of day based on the hour.

    This function returns a time of day category ('morning', 'afternoon', 'evening', 'night') based on the provided hour of the day.

    Parameters:
    -----------
    hr : int
        The hour of the day (0-23).

    Returns:
    --------
    str:
        The time of day category as a string.
    """
    if hr >= 6 and hr < 12 :
        return 'morning'
    elif hr >= 12 and hr <= 16 :
        return 'afternoon'
    elif hr > 16 and hr < 21 :
        return 'evening'
    else :
        return 'night'

def get_time_details(df):
    """
    Categorizes the time of day based on the hour.

    This function returns a time of day category ('morning', 'afternoon', 'evening', 'night') based on the provided hour of the day.

    Parameters:
    -----------
    hr : int
        The hour of the day (0-23).

    Returns:
    --------
    str:
        The time of day category as a string.
    """
    df['time_hour'] = df['time'].dt.hour
    return df

if __name__ == "__main__" :
    # df = load_file("data/cleaned_watch_history.csv")
    df = preprocess_data("data/cleaned_watch_history.csv")
    print(df.shape)










