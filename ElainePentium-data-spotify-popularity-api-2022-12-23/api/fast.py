import os
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
joblib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../assets'))
app.state.pipelined_model = joblib.load(f"{joblib_dir}/model.joblib")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# define a root `/` endpoint
@app.get("/")
def index():
    return {"Status": "Up and running"}


# Implement a /predict endpoint
@app.get("/predict")
def predict(acousticness, danceability, duration_ms, energy, explicit, id, instrumentalness, key,
            liveness, loudness, mode, name, release_date, speechiness, tempo, valence, artist):

    X_pred = pd.DataFrame({
        'acousticness':float(acousticness),
        'danceability':float(danceability),
        'duration_ms':int(duration_ms),
        'energy':float(energy),
        'explicit':int(explicit),
        'id':str(id),
        'instrumentalness':float(instrumentalness),
        'key':int(key),
        'liveness':float(liveness),
        'loudness':float(loudness),
        'mode':int(mode),
        'name':str(name),
        'release_date':str(release_date),
        'speechiness':float(speechiness),
        'tempo':float(tempo),
        'valence':float(valence),
        'artist':str(artist)
    }, index=[0])

    # return "COUCOU"

    y_pred = int(app.state.pipelined_model.predict(X_pred))

    return {"artist": artist,
            "name": name,
            "popularity": y_pred
            }
