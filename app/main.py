from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import requests
from dotenv import load_dotenv
import random
import chromadb  # Example VectorDB (You can use FAISS or Pinecone alternatively)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
# Load .env file
load_dotenv()

app = FastAPI()

# CORS setup for frontend (React Native, web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# YouTube API Key
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Initialize VectorDB (Chroma Example)
# chroma_client = chromadb.Client()
# collection = chroma_client.get_or_create_collection("textbook")
persist_dir = "chromadb_store"
embedding_func = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=persist_dir, embedding_function=embedding_func)


# YouTube search function
def search_youtube(query, max_results=3):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "key": YOUTUBE_API_KEY,
        "maxResults": max_results
    }
    response = requests.get(url, params=params)
    results = response.json()

    video_ids = [item["id"]["videoId"] for item in results.get("items", [])]
    return video_ids


# General Mode: Gemini + YouTube
def general_mode_response(question):
    student_prompt = (
        "You are a helpful AI tutor for school students. "
        "Respond to the following question using the Socratic method. "
        "Your answer should be conversational, ask follow-up questions, encourage thinking, "
        "and be simple and age-appropriate.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(student_prompt)
    answer = response.text.strip()

    video_ids = search_youtube(question)
    selected_video = random.choice(video_ids) if video_ids else "dQw4w9WgXcQ"

    youtube_iframe = (
        f'<iframe width="300" height="200" src="https://www.youtube.com/embed/{selected_video}" frameborder="0"></iframe>'
    )

    return {"answer": answer, "youtube": youtube_iframe}


# Textbook Mode: RAG + Gemini + YouTube
def textbook_mode_response(question):
    results = db.similarity_search(question, k=3)    
    if not results:
        return {
            "answer": "Sorry, I couldn't find relevant information in your textbook. Would you like a general answer instead?",
            "youtube": ""
        }
    context = " ".join([doc.page_content for doc in results])

    prompt = (
        "You are a helpful AI tutor for school students. "
        "Use ONLY the following textbook context to answer. "
        "Respond using the Socratic method, ask follow-up questions, encourage thinking, "
        "and be simple and age-appropriate.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    answer = response.text.strip()

    video_ids = search_youtube(question)
    selected_video = random.choice(video_ids) if video_ids else "dQw4w9WgXcQ"

    youtube_iframe = (
        f'<iframe width="300" height="200" src="https://www.youtube.com/embed/{selected_video}" frameborder="0"></iframe>'
    )

    return {"answer": answer, "youtube": youtube_iframe}

# API Route
@app.get("/ask")
def ask(question: str, mode: str = "general"):
    if mode == "general":
        return general_mode_response(question)
    elif mode == "textbook":
        return textbook_mode_response(question)
    else:
        return {"error": "Invalid mode"}
