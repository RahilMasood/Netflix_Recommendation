# Movie/Show Recommendation System

This is a Python-based movie recommendation system that provides similar movie and TV show recommendations based on a given movie or TV show title. The system leverages Natural Language Processing (NLP) techniques like **TF-IDF** and **Cosine Similarity** to generate recommendations. It uses data from a Netflix-like dataset to provide recommendations for movies and TV shows.

---

## Features

- **Recommends movies and TV shows** similar to a given movie or TV show.
- Utilizes **TF-IDF Vectorizer** and **Cosine Similarity** to compute similarity between titles based on multiple features (e.g., type, director, cast, rating, description, etc.).
- Provides top 10 similar results (both movies and TV shows).
- Cleans and preprocesses the dataset for effective feature extraction and recommendation.
- Filters recommendations based on a **similarity threshold** to provide more relevant results.

---

## Dataset

The dataset used in this project is a CSV file that contains information about movies and TV shows, including the following columns:

- **title**: Name of the movie or TV show.
- **type**: Type of content (either "Movie" or "TV Show").
- **director**: Director of the movie or TV show.
- **cast**: Cast members.
- **rating**: Rating of the movie or TV show.
- **listed_in**: Categories/genres the movie or TV show belongs to.
- **description**: A short description of the movie or TV show.
- **release_year**: Year the movie or TV show was released.
- **duration**: Duration of the movie (in minutes) or TV show (in seasons/episodes).
- **country**: Country of origin.
