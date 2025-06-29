from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import requests
from dotenv import load_dotenv
import random

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

# Configure Gemini API with your key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# YouTube API Key from .env
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Function to search YouTube
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

    video_ids = []
    for item in results.get("items", []):
        video_ids.append(item["id"]["videoId"])

    return video_ids

@app.get("/ask")
def ask(question: str):
    # Socratic-style prompt for Gemini
    student_prompt = (
        "You are a helpful AI tutor for school students. "
        "Respond to the following question using the Socratic method. "
        "Your answer should be conversational, ask follow-up questions, encourage thinking, "
        "and be simple and age-appropriate.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    # Search YouTube
    video_ids = search_youtube(question)

    if not video_ids:
        video_ids = ["dQw4w9WgXcQ"]  # Fallback video

    selected_video = random.choice(video_ids)
    youtube_iframe = (
        f'<iframe width="300" height="200" '
        f'src="https://www.youtube.com/embed/{selected_video}" frameborder="0"></iframe>'
    )

    # Generate answer with Gemini
    model = genai.GenerativeModel("gemini-1.5-pro")

    try:
        response = model.generate_content(student_prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    return {"answer": answer, "youtube": youtube_iframe}
