from fastapi import FastAPI, HTTPException,Body
from fastapi.responses import JSONResponse
from google.cloud import firestore
from pydantic import BaseModel
from typing import List
from functions import *
import random

app = FastAPI()

class RecommendationRequest(BaseModel):
    user_id: str
    n_recommendations: int = 5
    #input_songs: List[str]
    #top_n: int = 10

@app.post("/recommendForUser")
async def recommend_songs(request: RecommendationRequest):
    try:
        user_id = request.user_id
        n_recommendations = request.n_recommendations
        recommendations = recommend_songs_for_user(model_knn, pivot_table, user_id, n_recommendations)

        return {"user_id": user_id, "recommendations": recommendations}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Initialize Firestore client
db = firestore.Client.from_service_account_json("bardbeats-8f621-firebase-adminsdk-p44ov-773f1a6c29.json")

# Replace with your recommendation function
def calculate_recommendations(user_id: str, playlist: list, top_n=10):
    input_songs = playlist 
    song_indices = [data.index[data['song'] == song_id].tolist()[0] for song_id in input_songs if song_id in data['song'].values]
    
    # Aggregate the similarities of input songs with all songs
    aggregate_sim_scores = sum(cosine_similarity(tfidf_matrix[song_idx], tfidf_matrix) for song_idx in song_indices)

    # Flatten the similarity scores array and get top N indices
    sim_scores_flattened = aggregate_sim_scores.flatten()
    recommended_song_indices = sim_scores_flattened.argsort()[-top_n-len(input_songs):-len(input_songs)][::-1]
    
    # Get recommended song IDs excluding the input songs
    recommended_song_ids = [data.iloc[idx]['song'] for idx in recommended_song_indices if data.iloc[idx]['song'] not in input_songs]

    return recommended_song_ids

# Endpoint to calculate and store recommendations
@app.post("/calculate_recommendations/{user_id}")
async def calculate_and_store_recommendations(user_id: str):
    try:
        # Fetch user's collection from Firestore
        user_doc_ref = db.collection("users").document(user_id)
        user_doc = user_doc_ref.get()

        if user_doc.exists:
            # Fetch the user's playlist from Firestore
            user_data = user_doc.to_dict()
            if "playlist" in user_data:
                playlist_data = user_data["playlist"]
                playlist = [song_id for song_id, value in playlist_data.items() if value == 1]

                # Calculate recommendations
                recommendations = calculate_recommendations(user_id, playlist)

                # Store recommendations back in Firestore
                user_doc_ref.update({"recommendations": recommendations})

                return {"user_id": user_id, "recommendations": recommendations}
            else:
                raise HTTPException(status_code=404, detail="Playlist not found in user data")
        else:
            raise HTTPException(status_code=404, detail="User not found")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/get_all_users")
async def get_all_users():
    try:
        # Reference to the 'users' collection
        users_ref = db.collection("users")
        print(users_ref)
        # Fetch all user documents
        users_docs = users_ref.stream()

        users = []
        for user_doc in users_docs:
            user_data = user_doc.to_dict()
            user_data["id"] = user_doc.id  # Optionally include the document ID
            users.append(user_data)

        return {"users": users}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/write_random_valhalla")
async def write_random_valhalla():
    try:
        # Reference to the collection (e.g., 'data')
        collection_ref = db.collection("users")

        # Generate a random value (customize as needed)
        random_value = random.randint(1, 100)  # Example: random integer between 1 and 100

        # Create a new document with the random value under the key 'valhalla'
        doc_ref = collection_ref.add({"valhalla": random_value})

        # Fetch the generated document ID
        document_id = doc_ref[1].id

        return {"document_id": document_id, "valhalla": random_value}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    
    
@app.post("/add_song_to_playlist/{user_id}")
async def add_song_to_playlist(user_id: str, song_id: str = Body(..., embed=True)):
    try:
        # Reference to the user's document
        user_doc_ref = db.collection("users").document(user_id)
        user_doc = user_doc_ref.get()

        if user_doc.exists:
            # Fetch the user's playlist
            user_data = user_doc.to_dict()
            playlist = user_data.get("playlist", {})

            # Update the playlist with the new song
            # Assuming a simple structure where the value is a count of how many times the song is added
            playlist[song_id] = playlist.get(song_id, 0) + 1

            # Store the updated playlist back in Firestore
            user_doc_ref.update({"playlist": playlist})

            return {"user_id": user_id, "song_id": song_id, "playlist": playlist}
        else:
            raise HTTPException(status_code=404, detail="User not found")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))