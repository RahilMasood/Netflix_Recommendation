import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Read data
netflix_data = pd.read_csv(r"C:\Users\shez8\Desktop\RAHIL\Mini Projects\Movie Recommender\netflix_data.csv")
netflix_data.fillna('', inplace=True)

# Preprocess data
netflix_data['description'] = netflix_data['description'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower())

# Feature engineering
new_data = netflix_data[['title', 'type', 'director', 'cast', 'rating', 'listed_in', 'description']]
new_data.set_index('title', inplace=True)

class TextCleaner:
    def clean_text(self, text):
        text = ' '.join(set(text.split(','))).lower()  # Remove duplicates (comma-separated) and lowercase
        text = text.replace(' ', '')  # Remove spaces
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        return text

cleaner = TextCleaner()
new_data['type'] = new_data['type'].apply(cleaner.clean_text)
new_data['director'] = new_data['director'].apply(cleaner.clean_text)
new_data['cast'] = new_data['cast'].apply(cleaner.clean_text)
new_data['rating'] = new_data['rating'].apply(cleaner.clean_text)
new_data['listed_in'] = new_data['listed_in'].apply(cleaner.clean_text)
new_data['description'] = new_data['description'].apply(cleaner.clean_text)

# Combine features into a single string for each movie
new_data['BoW'] = new_data[['type', 'director', 'cast', 'rating', 'listed_in', 'description']].apply(lambda row: ' '.join(row), axis=1)

# TF-IDF and Cosine Similarity
tfid = TfidfVectorizer()
tfid_matrix = tfid.fit_transform(new_data['BoW'])
cosine_sim = cosine_similarity(tfid_matrix, tfid_matrix)

np.save('tfidf_matrix.npy', tfid_matrix)
np.save('cosine_sim_matrix.npy', cosine_sim)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfid, f)

final_data = netflix_data[['title', 'type']]
final_data.to_csv('movie_data.csv', index=False)

class FlixHub:
    def __init__(self, df, cosine_sim):
        self.df = df
        self.cosine_sim = cosine_sim
    
    def recommendation(self, title, total_result=5, threshold=0.5):
        idx = self.find_id(title)
        if idx == -1:
            return "Movie not found."
        
        # Calculate similarities and filter based on the threshold
        similarities = self.cosine_sim[idx]
        self.df['similarity'] = similarities
        sorted_df = self.df.sort_values(by='similarity', ascending=False)
        
        # Filter out movies and TV shows below the threshold
        sorted_df = sorted_df[sorted_df['similarity'] >= threshold]
        
        # Get movies and TV shows separately
        movies = sorted_df[sorted_df['type'] == 'Movie']['title'].head(total_result)
        tv_shows = sorted_df[sorted_df['type'] == 'TV Show']['title'].head(total_result)
        
        # Format the results
        similar_movies = [f"{i+1}. {movie}" for i, movie in enumerate(movies)]
        similar_tv_shows = [f"{i+1}. {tv_show}" for i, tv_show in enumerate(tv_shows)]
        
        return similar_movies, similar_tv_shows

    def find_id(self, name):
        # Search for movie by title
        idx = self.df[self.df['title'].str.contains(name, case=False, na=False)].index
        return idx[0] if len(idx) > 0 else -1

# Example usage:
flix_hub = FlixHub(final_data, cosine_sim)
movies, tv_shows = flix_hub.recommendation('13 Reasons Why', total_result=10, threshold=0.5)

print('Similar Movie(s) list:')
for movie in movies:
    print(movie)

print('\nSimilar TV Show(s) list:')
for tv_show in tv_shows:
    print(tv_show)
