import pandas as pd
import streamlit as st
import numpy as np
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from streamlit_tags import st_tags

# Load the DataFrame
df1 = pd.read_csv('/Anime_data.csv')

# Define the function to search for titles and return matching rows
def search_titles(df, keyword, column='title', limit=30):
    results = process.extract(keyword, df[column], limit=limit)
    matched_titles = [result[0] for result in results]
    return df[df[column].isin(matched_titles)]

# Define the recommender function
def recommend_titles(df, title, top_n=20):
    # Categorical features to use
    cat_features = ['title', 'genres', 'themes', 'studios', 'source', 'demographic', 'type']
    # Numerical features to use
    num_features = ['popularity', 'score', 'position', 'members']
    # Define weights for each feature category
    weights = {
        'title': 10.5,
        'genres': 6.0,
        'themes': 6.5,
        'studios': 2.4,
        'source': 1.2,
        'demographic': 1.0,
        'type': 1.0,
        'popularity': 5.7,
        'score': 4.5,
        'position': 3.2,
        'members': 3.0
    }

    # Combine categorical features into a single string per row
    df['combined_features'] = df[cat_features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

    # Apply weights to the categorical features
    weighted_tfidf_matrix = tfidf_matrix * weights['title']

    # Compute the feature matrices individually and add the weighted results
    for feature, weight in zip(cat_features[1:], [weights[feature] for feature in cat_features[1:]]):
        feature_matrix = tfidf_vectorizer.transform(df[feature])
        weighted_tfidf_matrix += feature_matrix * weight

    # Apply weights to the numerical features
    scaler = StandardScaler()
    scaled_num_features = scaler.fit_transform(df[num_features])
    weighted_num_features = scaled_num_features * np.array([weights[feature] for feature in num_features])
    
    # Combine weighted features
    combined_features = np.hstack((weighted_tfidf_matrix.toarray(), weighted_num_features))
    
    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(combined_features)
    
    # Get the index of the input title
    idx = df.index[df['title'] == title][0]
    
    # Get similarity scores for all titles
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the titles based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the most similar titles
    sim_scores = sim_scores[1:top_n+1]  # Exclude the input title itself
    
    # Get the titles of the most similar entries
    title_indices = [i[0] for i in sim_scores]
    
    return df.iloc[title_indices]

# Main Streamlit app
def main():
    st.title('Anime Title Search Engine')

    # Input for searching titles with suggestions
    search_term = st_tags(
        label='Enter a title to search:',
        text='Press enter to add more',
        value='',
        suggestions=df1['title'].tolist(),
        maxtags=1,
        key='1'
    )

    if search_term:
        selected_title = search_term[0]
        st.subheader('Selected Title Details:')
        st.write(df1[df1['title'] == selected_title])

        # Display recommended titles
        st.subheader('Recommended Titles:')
        recommendations = recommend_titles(df1, selected_title, top_n=20)
        st.write(recommendations[['title', 'genres', 'themes', 'score', 'position', 'popularity', 'studios', 'source', 'type', 'demographic', 'members']])

if __name__ == '__main__':
    main()
