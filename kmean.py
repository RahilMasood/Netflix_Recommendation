import string
import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re

netflix_data = pd.read_csv(r"C:\Users\shez8\Desktop\RAHIL\Mini Projects\Movie Recommender\netflix_data.csv")
netflix_data.fillna('', inplace=True)

titles = ' '.join(netflix_data['title'].values)
#title_wordcloud = WordCloud(background_color='black', colormap='Reds').generate(titles)

selected_data = netflix_data[['title', 'type', 'director', 'cast', 'rating', 'listed_in', 'description']]
selected_data.set_index('title', inplace=True)

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

selected_data['BoW'] = selected_data.apply(lambda row: ' '.join(row.dropna().values), axis=1)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(selected_data['BoW'])

num_clusters = 10  # Choose optimal k based on domain knowledge or methods like Elbow Method
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)

final_dataset = netflix_data[['title', 'type']]
final_dataset['cluster'] = clusters

class MoviesKMeans:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_recommendations(self, query_title, max_results=5):
        query_index = self.find_title_index(query_title)
        if query_index == -1:
            return [], []

        query_cluster = self.dataset.loc[query_index, 'cluster']

        cluster_data = self.dataset[self.dataset['cluster'] == query_cluster]

        movies = cluster_data[cluster_data['type'] == 'Movie']['title'].head(max_results)
        tv_shows = cluster_data[cluster_data['type'] == 'TV Show']['title'].head(max_results)

        movie_list = [f"{i + 1}. {title}" for i, title in enumerate(movies)]
        show_list = [f"{i + 1}. {title}" for i, title in enumerate(tv_shows)]

        return movie_list, show_list

    def find_title_index(self, title):
        for idx, movie_title in enumerate(self.dataset['title']):
            if re.search(title, movie_title, re.IGNORECASE):
                return idx
        return -1

movie_name = input("Enter a movie or TV show title: ")

movie_kmeans = MoviesKMeans(final_dataset)
recommended_movies, recommended_shows = movie_kmeans.get_recommendations(movie_name, max_results=10)

print('\nSimilar Movie(s) list:')
for movie in recommended_movies:
    print(movie)

print('\nSimilar TV_show(s) list:')
for show in recommended_shows:
    print(show)
