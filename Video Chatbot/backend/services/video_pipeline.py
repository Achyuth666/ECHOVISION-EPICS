import os
import os
import sys
import shutil
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
from video_captioning_src.video_to_frames.video_to_frame import extract_frames_from_video
from video_captioning_src.image_captioning.Image_Captioning import ImageCaptioning
from video_chatbot.src.ingest import create_vector_db

def process_video(video_path, frames_dir, transcript_dir):
    if os.path.exists(transcript_dir):
        print("Clearing old transcript_dir...")
        shutil.rmtree(transcript_dir)

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(transcript_dir, exist_ok=True)

    # 1. Extract frames
    extract_frames_from_video(video_path, frames_dir, interval=1)

    # 2. Generate captions
    captioner = ImageCaptioning()
    captions = captioner.generate_captions(frames_dir)

    # 3. Save transcript
    transcript_path = os.path.join(transcript_dir, "video_analysis.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        for line in captions:
            f.write(line + "\n")

    # 4. Create vector DB
    create_vector_db()

    return transcript_path