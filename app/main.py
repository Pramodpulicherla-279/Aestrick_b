from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import requests
from dotenv import load_dotenv
import random
from chroma_db import retriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
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
#CRC setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.getenv("GEMINI_API_KEY"))
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# memory = ConversationBufferMemory()
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True, 
    output_key='answer'  # <-- Add this line
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer"  # <-- Add this line
)
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

def build_socratic_prompt(question, context=None):
    base = (
        "You are a patient, helpful AI tutor for school students. "
        "Answer the student's question clearly, simply, and in an age-appropriate way. "
        "After your answer, ask exactly ONE short follow-up question to encourage the student to think, "
        "but make sure your follow-up is strictly related to the student's original question or topic. "
        "Do not introduce unrelated topics or ideas."
    )
    
    if context:
        base += (
            "\n\nUse ONLY the following textbook context to answer. "
            "If the context does not contain the answer, say politely that you don't know the answer based on the textbook, "
            "and suggest switching to general knowledge if the student wants."
            f"\n\nContext:\n{context}"
        )
        
    base += f"\n\nStudent's Question: {question}\n\nAnswer:"
    
    return base

# General Mode: Gemini + YouTube
def general_mode_response(question):
    prompt = build_socratic_prompt(question)
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    answer = response.text.strip()
    video_ids = search_youtube(question)
    selected_video = random.choice(video_ids) if video_ids else "dQw4w9WgXcQ"
    youtube_iframe = (
        f'<iframe width="300" height="200" src="https://www.youtube.com/embed/{selected_video}" frameborder="0"></iframe>'
    )
    return {"answer": answer, "youtube": youtube_iframe}

# Textbook Mode: RAG + Gemini + YouTube
# def textbook_mode_response(question):
#     results = db.similarity_search(question, k=3)
#     if not results:
#         # No relevant context, switch to general mode
#         return {
#             "answer": "This is an irrelevant question for textbook knowledge, I'm switching to general mode.",
#             **general_mode_response(question)
#         }
#     context = " ".join([doc.page_content for doc in results])
#     # print(results)

#     prompt = (
#         "You are a helpful AI tutor for school students. "
#         "Use ONLY the following textbook context to answer. "
#         "If the context does not contain the answer, say you don't know. "
#         "Respond using the Socratic method, ask follow-up questions, encourage thinking, "
#         "and be simple and age-appropriate.\n\n"
#         f"Context:\n{context}\n\n"
#         f"Question: {question}\n"
#         "Answer:"
#     )

#     model = genai.GenerativeModel("gemini-2.5-flash")
#     response = model.generate_content(prompt)
#     answer = response.text.strip()

#     video_ids = search_youtube(question)
#     selected_video = random.choice(video_ids) if video_ids else "dQw4w9WgXcQ"

#     youtube_iframe = (
#         f'<iframe width="300" height="200" src="https://www.youtube.com/embed/{selected_video}" frameborder="0"></iframe>'
#     )

#     return {"answer": answer, "youtube": youtube_iframe}

# def textbook_mode_response(question):
#     try:
#         results = db.similarity_search_with_score(question, k=3)
#         SIMILARITY_THRESHOLD = 1
#         filtered = [doc for doc, score in results if score < SIMILARITY_THRESHOLD]
#     except Exception:
#         filtered = db.similarity_search(question, k=3)
#     if not filtered:
#         general = general_mode_response(question)
#         general["answer"] = (
#             "This is an irrelevant question for textbook knowledge, I'm switching to general mode.<br><br>"
#             + general["answer"]
#         )
#         return general
#     context = " ".join([doc.page_content for doc in filtered])
#     prompt = build_socratic_prompt(question, context)
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     response = model.generate_content(prompt)
#     answer = response.text.strip()
#     video_ids = search_youtube(question)
#     selected_video = random.choice(video_ids) if video_ids else "dQw4w9WgXcQ"
#     youtube_iframe = (
#         f'<iframe width="300" height="200" src="https://www.youtube.com/embed/{selected_video}" frameborder="0"></iframe>'
#     )
#     return {"answer": answer, "youtube": youtube_iframe}

def textbook_mode_response(question):
    result = qa_chain.invoke({"question": question, "chat_history": []})
    # result = qa_chain({"question": question})
    
    answer = result["answer"]
    sources = result.get("source_documents", [])
    
    # If no source documents, fallback to general mode
    if not sources:
        general = general_mode_response(question)
        general["answer"] = (
            "This seems unrelated to the textbook. Switching to general mode.<br><br>" +
            general["answer"]
        )
        return general

    video_ids = search_youtube(question)
    selected_video = random.choice(video_ids) if video_ids else "dQw4w9WgXcQ"
    
    youtube_iframe = (
        f'<iframe width="300" height="200" src="https://www.youtube.com/embed/{selected_video}" frameborder="0"></iframe>'
    )
    
    return {"answer": answer, "youtube": youtube_iframe}

    # print(results)

    # prompt = (
    #     "You are a helpful AI tutor for school students. "
    #     "Use ONLY the following textbook context to answer. "
    #     "If the context does not contain the answer, say you don't know. "
    #     "Respond using the Socratic method, ask follow-up questions, encourage thinking, "
    #     "and be simple and age-appropriate.\n\n"
    #     f"Context:\n{context}\n\n"
    #     f"Question: {question}\n"
    #     "Answer:"
    # )

    # model = genai.GenerativeModel("gemini-1.5-flash")
    # response = model.generate_content(prompt)
    # answer = response.text.strip()

    # video_ids = search_youtube(question)
    # selected_video = random.choice(video_ids) if video_ids else "dQw4w9WgXcQ"

    # youtube_iframe = (
    #     f'<iframe width="300" height="200" src="https://www.youtube.com/embed/{selected_video}" frameborder="0"></iframe>'
    # )

    # return {"answer": answer, "youtube": youtube_iframe}

# API Route
@app.get("/ask")
def ask(question: str, mode: str = "general"):
    if mode == "general":
        return general_mode_response(question)
    elif mode == "textbook":
        return textbook_mode_response(question)
    else:
        return {"error": "Invalid mode"}
