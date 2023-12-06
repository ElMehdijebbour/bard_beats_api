import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# Load the dataset
file_path = 'song_dataset.csv'  # Update the file path if needed
data = pd.read_csv(file_path)

# Preprocessing
## Remove duplicates if any
data = data.drop_duplicates(['user', 'song'])

## Create a pivot table with users as rows and songs as columns
pivot_table = data.pivot(index='user', columns='song', values='play_count').fillna(0)

## Convert the pivot table to a sparse matrix
matrix = csr_matrix(pivot_table.values)

# Compute Cosine Similarity
cosine_sim = cosine_similarity(matrix)
 
# Building the Model - Using K-Nearest Neighbors
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(matrix)

# Function 1 to recommend songs for a user
def recommend_songs_for_user(model, pivot_table, user_id, n_recommendations):
    if user_id not in pivot_table.index:
        return "User ID not found in the dataset."

    user_idx = pivot_table.index.get_loc(user_id)
    distances, indices = model.kneighbors(pivot_table.iloc[user_idx, :].values.reshape(1, -1), n_neighbors=20)
    
    # Get the songs listened to by the user
    listened_songs = set(pivot_table.columns[pivot_table.iloc[user_idx].to_numpy().nonzero()[0]].tolist())

    recommend_list = set()
    for idx in indices.flatten():
        if idx == user_idx:
            continue  # skip the user itself
        # Add songs listened by similar users, excluding those already listened by the user
        user_songs = set(pivot_table.columns[pivot_table.iloc[idx].to_numpy().nonzero()[0]].tolist())
        recommend_list.update(user_songs)

    # Remove already listened songs and limit the number of recommendations
    recommend_list.difference_update(listened_songs)
    recommend_list = list(recommend_list)[:n_recommendations]

    return recommend_list


#################################################


# Function 2 to Recommend Songs
def recommend_songs_collaborative_filtering(input_songs, pivot_table, cosine_sim, top_n=10):
    # Create a pseudo-user vector
    pseudo_user = pd.Series(0, index=pivot_table.columns)
    for song in input_songs:
        if song in pseudo_user.index:
            pseudo_user[song] = 1

    # Find similar users
    pseudo_user_matrix = csr_matrix(pseudo_user.values.reshape(1, -1))
    sim_scores = cosine_similarity(pseudo_user_matrix, matrix).flatten()
    similar_users = sim_scores.argsort()[::-1][1:]  # Excluding the pseudo-user itself

    # Aggregate recommendations from similar users
    song_recs = {}
    for user_idx in similar_users[:20]:  # Consider top 20 similar users
        user_songs = pivot_table.columns[(pivot_table.iloc[user_idx] > 0)].tolist()
        for song in user_songs:
            if song not in input_songs:
                song_recs[song] = song_recs.get(song, 0) + sim_scores[user_idx]

    recommended_songs = sorted(song_recs, key=song_recs.get, reverse=True)[:top_n]
    return recommended_songs



#################################################

## Combine text features for content-based filtering
data['combined_features'] = data['title'] + ' ' + data['artist_name'] + ' ' + data['release']

# Content-based Features using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['combined_features'])

# Function 3 to Recommend Songs
def recommend_songs_content_based(input_songs, data, tfidf_matrix, top_n=10):
    song_indices = [data.index[data['song'] == song_id].tolist()[0] for song_id in input_songs if song_id in data['song'].values]
    
    # Aggregate the similarities of input songs with all songs
    aggregate_sim_scores = sum(cosine_similarity(tfidf_matrix[song_idx], tfidf_matrix) for song_idx in song_indices)

    # Flatten the similarity scores array and get top N indices
    sim_scores_flattened = aggregate_sim_scores.flatten()
    recommended_song_indices = sim_scores_flattened.argsort()[-top_n-len(input_songs):-len(input_songs)][::-1]
    
    # Get recommended song IDs excluding the input songs
    recommended_song_ids = [data.iloc[idx]['song'] for idx in recommended_song_indices if data.iloc[idx]['song'] not in input_songs]

    return recommended_song_ids