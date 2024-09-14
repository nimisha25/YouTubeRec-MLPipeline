import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from sklearn import preprocessing
import gensim
import gensim.downloader as api
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np
from sklearn.preprocessing import StandardScaler
#model = api.load('word2vec-google-news-300')
#model.save('word2vec-google-news-300.model')
model = KeyedVectors.load('word2vec-google-news-300.model')

def load_file (filename) :
    df = pd.read_csv(filename)
    return df

def remove_stopwords(words) :
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []
    for w in words:
        if w == 'watched' :
            continue
        if w not in stop_words:
            filtered_sentence.append(w)
    
    return filtered_sentence

def normalise_features(values) :
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(values)
    return x_scaled
    # x_array = np.array(df['view_count'])
    # normalized_arr = preprocessing.normalize([x_array])
    # df = pd.replace(df['view_count'], normalise_features, inplace = True)
    # return df

def word2vec_embeddings(words) :
    embeddings = [model[word] for word in words if word in model]
    # Return the average of the embeddings
    if embeddings:  # axis = 0 means along the column
        avg_embedding = np.mean(embeddings, axis=0)
    else:
        avg_embedding = np.zeros(model.vector_size)  # Return a vector of zeros if no embeddings found
    
    return avg_embedding

def process_tags(tags) :
    tokenizer = RegexpTokenizer(r'\w+')
    if not tags or len(tags) == 0:
        return np.zeros(model.vector_size)  # Return a zero vector if no tags are present

    all_embeddings = []
    for phrase in tags:
        # Tokenize and remove stopwords from the phrase
        split = tokenizer.tokenize(phrase)
        split = remove_stopwords(split)
        # Get the word embeddings for the phrase
        embedding = word2vec_embeddings(split)
        all_embeddings.append(embedding)

    # Return the average of all phrase embeddings
    return np.mean(all_embeddings, axis=0) if all_embeddings else np.zeros(model.vector_size)

def preprocess_data(df) :

    # # normalise features 'view_count' and 'like_count'
    # df['view_count'] = normalise_features(df[['view_count']])
    # df['like_count'] = normalise_features(df[['like_count']])

    # get word2vec embeddings for title
    tokenizer = RegexpTokenizer(r'\w+')
    df['title'] = df['title'].apply(lambda x : x.lower())
    df['title'] = df['title'].apply(lambda x : tokenizer.tokenize(x))
    df['title'] = df['title'].apply(lambda x : remove_stopwords(x))

    # get word2vec embeddings for tags
    df['tags'] = df['tags'].apply(lambda x : [] if x is None else x.lower())
    df['title_avg_embedding'] = df['title'].apply(lambda x : word2vec_embeddings(x))
    df['tag_embedding'] = df['tags'].apply(lambda x : process_tags(x))
    
    # processing 'time'
    df = df.dropna(subset=['time'])
    # nan_count = df['time'].isna().sum()
    # print("nan count  is: " ,nan_count)
    # nan_rows = df[df['time'].isna()]
    df['time'] = pd.to_datetime(df['time'])
    df['local_time'] = df['time'].dt.tz_convert('US/Eastern')
    df = get_time_details(df)

    # one hot encoding the day category
    df['day_category'] = df['time_hour'].apply(ret_day_category)
    df = pd.get_dummies(df, columns = ['day_category'])

    # df with all the processed features
    df_processed = combine_features(df)
    df_processed.columns = df_processed.columns.astype(str)

    scaler = StandardScaler()
    df_processed.iloc[:, :] = scaler.fit_transform(df_processed)

    return df_processed

def combine_features(df) :
    title_emb_df = pd.DataFrame(df['title_avg_embedding'].tolist(), index=df.index)
    tag_emb_df = pd.DataFrame(df['tag_embedding'].tolist(), index=df.index)

    numeric_features = df[['view_count', 'like_count']]
    time_features = df[['time_hour']]
    one_hot_features = df[['day_category_morning', 'day_category_afternoon', 'day_category_evening', 'day_category_night']]
    final_feature_set = pd.concat([title_emb_df, tag_emb_df, numeric_features, time_features, one_hot_features], axis=1)  
    return final_feature_set
    

def ret_day_category(hr) :
    if hr >= 6 and hr < 12 :
        return 'morning'
    elif hr >= 12 and hr <= 16 :
        return 'afternoon'
    elif hr > 16 and hr < 21 :
        return 'evening'
    else :
        return 'night'

def get_time_details(df):
    df['time_hour'] = df['time'].dt.hour
    return df

if __name__ == "__main__" :
    df = load_file("data/cleaned_watch_history.csv")
    df = preprocess_data(df)
    df.to_csv("data/final_feature_set.csv", index=False)
    print(df.shape)










