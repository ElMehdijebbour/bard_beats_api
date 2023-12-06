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



## Combine text features for content-based filtering
data['combined_features'] = data['title'] + ' ' + data['artist_name'] + ' ' + data['release']

# Content-based Features using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['combined_features'])
