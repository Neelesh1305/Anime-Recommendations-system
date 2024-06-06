# Anime Recommender system:
This project involves building an Anime Recommender System using data from the MyAnimeList Dataset. The dataset contains detailed information about various anime series, including their popularity, score, genres, themes, and more. The recommender system leverages the TF-IDF vectorizer and cosine similarity to provide personalized anime recommendations.

## Data Source:
The dataset used for this project is sourced from Kaggle, specifically the "MyAnimeList Dataset" by ajaxlima. The dataset can be accessed here.

## Features:
The dataset includes the following features:

position: Position in the dataset.
 
title: Name of the anime.

episodes: Number of episodes.

release_date: Release date of the anime.

members: Number of people in that particular anime's community.

score: Score given by the audience.

details: Source details about the anime and the relevant links.

broadcast: Broadcast information.

studios: Studios that produced the anime.

source: Source material of the anime.

genres: Genres of the anime.

themes: Themes present in the anime.

demographic: Target demographic of the anime.

popularity: Popularity ranking of the anime.

## Feature Extraction and Engineering:

Relevant features such as title, genres, themes, studios, source, demographic, type, popularity, score, position, and members were combined into a single string for each anime entry.
Numerical features were standardized to ensure uniformity in the recommendation calculations.

## Recommendation system

The combined categorical features were vectorized using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to convert text data into numerical form. This process helped in highlighting the importance of each feature while minimizing the influence of commonly occurring terms.
Cosine similarity was used to measure the similarity between anime entries based on their vectorized feature representations. This metric helps in identifying how similar two anime entries are in terms of their features.
Each feature was assigned a weight based on its importance in generating accurate recommendations. The weighted combination of categorical and numerical features ensured that the most relevant aspects of each anime were considered.

## Implementation and Outputs

The system takes an input anime title and provides the top 20 most similar anime recommendations based on the cosine similarity scores. The recommendations are displayed along with their relevant details such as title, genres, themes, score, position, popularity, studios, source, type, demographic, and members.

A user-friendly Streamlit web application was developed to interact with the recommender system. Users can enter an anime title, receive real-time suggestions with auto-complete functionality, and view the top 20 recommended anime along with their details.

This project successfully demonstrates the use of natural language processing techniques and machine learning algorithms to create a personalized anime recommender system, providing a valuable tool for anime enthusiasts to discover new shows based on their preferences.

