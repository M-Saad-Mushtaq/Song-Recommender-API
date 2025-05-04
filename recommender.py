import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

class SongRecommender:
    def __init__(self):
        self.songs = pd.read_csv("songs_with_emotions.csv")
        
        # Load and process documents
        raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)
        self.db_songs = Chroma.from_documents(documents, OpenAIEmbeddings())

    def retrieve_semantic_recommendations(
            self,
            query: str,
            genre: str = None,
            tone: str = None,
            initial_top_k: int = 50,
            final_top_k: int = 16,
    ) -> pd.DataFrame:
        recs = self.db_songs.similarity_search(query, k=initial_top_k)
        songs_list = [rec.page_content.strip('"').split()[0] for rec in recs]
        song_recs = self.songs[self.songs["id"].isin(songs_list)].head(initial_top_k)

        if genre and genre != "All":
            song_recs = song_recs[song_recs["genre"] == genre].head(final_top_k)
        else:
            song_recs = song_recs.head(final_top_k)

        if tone == "Happy":
            song_recs.sort_values(by="joy", ascending=False, inplace=True)
        elif tone == "Surprising":
            song_recs.sort_values(by="surprise", ascending=False, inplace=True)
        elif tone == "Angry":
            song_recs.sort_values(by="anger", ascending=False, inplace=True)
        elif tone == "Suspenseful":
            song_recs.sort_values(by="fear", ascending=False, inplace=True)
        elif tone == "Sad":
            song_recs.sort_values(by="sadness", ascending=False, inplace=True)

        return song_recs

    def get_recommendations(
            self,
            query: str,
            genre: str = "All",
            tone: str = "All"
    ) -> list:
        recommendations = self.retrieve_semantic_recommendations(query, genre, tone)
        results = []

        for _, row in recommendations.iterrows():
            description = row["description"]
            truncated_desc_split = description.split()
            truncated_description = " ".join(truncated_desc_split[:30]) + "..."

            artists_split = row["artists"].split(";")
            if len(artists_split) == 2:
                artists_str = f"{artists_split[0]} and {artists_split[1]}"
            elif len(artists_split) > 2:
                artists_str = f"{', '.join(artists_split[:-1])}, and {artists_split[-1]}"
            else:
                artists_str = row["artists"]

            thumbnail = row.get("large_thumbnail", "cover-not-found.jpg") or "cover-not-found.jpg"

            results.append({
                "name": row["name"],
                "artists": artists_str,
                "description": truncated_description,
                "thumbnail": thumbnail,
                "genre": row["genre"],
                "tone": tone
            })
        return results