from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from recommender import SongRecommender

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = SongRecommender()

class RecommendationRequest(BaseModel):
    query: str
    genre: Optional[str] = "All"
    tone: Optional[str] = "All"

@app.get("/genres")
async def get_genres():
    return {"genres": ["All"] + sorted(recommender.songs["genre"].unique().tolist())}

@app.get("/tones")
async def get_tones():
    return {"tones": ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]}

@app.post("/recommend")
async def recommend_songs(request: RecommendationRequest):
    try:
        recommendations = recommender.get_recommendations(
            query=request.query,
            genre=request.genre,
            tone=request.tone
        )
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Song Recommender API is running!"}