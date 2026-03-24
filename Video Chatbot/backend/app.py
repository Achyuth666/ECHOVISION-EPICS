import os
import sys
from typing import Optional

# Add project root to PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil

from services.video_pipeline import process_video
from video_chatbot.src.chat import ChatBot

# -----------------------------
# App setup
# -----------------------------
app = FastAPI(title="Video RAG Backend")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DOCS_DIR = os.path.join(PROJECT_ROOT, "video_chatbot", "docs_dir")

os.makedirs(UPLOAD_DIR, exist_ok=True)

bot: Optional[ChatBot] = None  # global chatbot instance

# -----------------------------
# Upload endpoint
# -----------------------------
@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    global bot
    bot = None
    video_path = os.path.join(UPLOAD_DIR, video.filename)

    # Save uploaded video
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Process video (frames → captions → vector DB)
    process_video(video_path, FRAMES_DIR, DOCS_DIR)

    # Initialize chatbot ONCE
    if bot is None:
        bot = ChatBot()

    return {"status": "Video processed successfully"}

# -----------------------------
# Chat endpoint
# -----------------------------
@app.post("/chat")
async def chat(payload: dict):
    global bot

    if bot is None:
        raise HTTPException(
            status_code=400,
            detail="No video processed yet"
        )

    question = payload.get("question")
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Missing question"
        )

    try:
        response = bot.ask(question)

        # LlamaIndex-safe extraction
        if hasattr(response, "response"):
            answer_text = response.response
        else:
            answer_text = str(response)

        return {
            "answer": answer_text,
            "error": None
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "answer": "",
                "error": str(e)
            }
        )