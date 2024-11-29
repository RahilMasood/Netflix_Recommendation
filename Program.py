import string
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import re

warnings.filterwarnings('ignore')

# Load the Netflix dataset
netflix_data = pd.read_csv(r"C:\Users\shez8\Desktop\RAHIL\Mini Projects\Movie Recommender\netflix_data.csv")
netflix_data.fillna('', inplace=True)

# Analyzing data distributions
release_year_counts = netflix_data['release_year'].value_counts().sort_index()
content_type_counts = netflix_data['type'].value_counts()
top_countries = netflix_data['country'].value_counts().head(10)

# Extracting ratings and durations
rating_labels = list(netflix_data['rating'].value_counts().index)
rating_values = list(netflix_data['rating'].value_counts().values)

duration_labels = list(netflix_data['duration'].value_counts().index)
duration_values = list(netflix_data['duration'].value_counts().values)

# Generating word clouds for titles, descriptions, and genres
all_titles = ' '.join(netflix_data['title'].values)
title_wordcloud = WordCloud(background_color='black', colormap='Reds').generate(all_titles)

all_descriptions = ' '.join(netflix_data['description'].values)
description_wordcloud = WordCloud(background_color='black', colormap='Reds').generate(all_descriptions)

all_genres = ' '.join(netflix_data['listed_in'].values)
genre_wordcloud = WordCloud(background_color='black', colormap='Reds').generate(all_genres)

# Prepare the dataset for text processing
selected_data = netflix_data[['title', 'type', 'director', 'cast', 'rating', 'listed_in', 'description']]
selected_data.set_index('title', inplace=True)

# Helper class for text cleaning
class TextCleaner:
    def split_and_lower(self, text):
        unique_parts = set()
        for part in text.split(','):
            unique_parts.add(part.strip().lower())
        return ' '.join(unique_parts)

    def compact_text(self, text):
        return text.replace(' ', '').lower()

    def clean_punctuation(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(text.split())

    def process_text(self, text):
        text = self.split_and_lower(text)
        text = self.compact_text(text)
        text = self.clean_punctuation(text)
        return text

cleaner = TextCleaner()

selected_data['type'] = selected_data['type'].apply(cleaner.compact_text)
selected_data['director'] = selected_data['director'].apply(cleaner.split_and_lower)
selected_data['cast'] = selected_data['cast'].apply(cleaner.split_and_lower)
selected_data['rating'] = selected_data['rating'].apply(cleaner.compact_text)
selected_data['listed_in'] = selected_data['listed_in'].apply(cleaner.split_and_lower)
selected_data['description'] = selected_data['description'].apply(cleaner.clean_punctuation)

# Combine cleaned columns into a Bag of Words
selected_data['BoW'] = selected_data.apply(lambda row: ' '.join(row.dropna().values), axis=1)
selected_data = selected_data[['BoW']]

# TF-IDF vectorization and similarity calculation
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(selected_data['BoW'])
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save the vectorizer and similarity matrix for future use
np.save('tfidf_matrix.npy', tfidf_matrix)
np.save('cosine_sim_matrix.npy', similarity_matrix)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Prepare the final data for recommendations
final_dataset = netflix_data[['title', 'type']]
final_dataset.to_csv('movie_data.csv', index=False)

# Recommendation system class
class Movies:
    def __init__(self, dataset, similarity_matrix):
        self.dataset = dataset
        self.similarity_matrix = similarity_matrix

    def get_recommendations(self, query_title, max_results=5, similarity_threshold=0.5):
        index = self.find_title_index(query_title)
        self.dataset['similarity'] = self.similarity_matrix[index]
        sorted_dataset = self.dataset.sort_values(by='similarity', ascending=False)[1:max_results+1]

        movie_recs = sorted_dataset['title'][sorted_dataset['type'] == 'Movie']
        show_recs = sorted_dataset['title'][sorted_dataset['type'] == 'TV Show']

        movie_list = ['{}. {}'.format(i + 1, movie) for i, movie in enumerate(movie_recs)]
        show_list = ['{}. {}'.format(i + 1, show) for i, show in enumerate(show_recs)]

        return movie_list, show_list

    def find_title_index(self, title):
        for idx, movie_title in enumerate(self.dataset['title']):
            if re.search(title, movie_title, re.IGNORECASE):
                return idx
        return -1

# Initialize and test the recommendation system
movie_name = input("Enter a movie or TV show title: ")  # Prompt the user for input

movie = Movies(final_dataset, similarity_matrix)
recommended_movies, recommended_shows = movie.get_recommendations(movie_name, max_results=10, similarity_threshold=0.5)

print('\nSimilar Movie(s) list:')
for movie in recommended_movies:
    print(movie)

print('\nSimilar TV_show(s) list:')
for show in recommended_shows:
    print(show)
