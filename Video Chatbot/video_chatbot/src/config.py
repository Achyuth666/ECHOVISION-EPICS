import os

# Root Directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# slm model path
MODEL_PATH = os.path.join(ROOT_DIR, "models", "qwen2.5-1.5b-instruct-q4_k_m.gguf")

# Data Paths
DB_PATH = os.path.join(ROOT_DIR, "chroma_db")
DOCS_DIR = os.path.join(ROOT_DIR, "video_chatbot", "docs_dir")
TRANSCRIPT_FILE = os.path.join(DOCS_DIR, "video_analysis.txt")

# Settings
FRAME_INTERVAL = 1
IMAGE_QUALITY = 80
CONTEXT_WINDOW = 2048
MAX_OUTPUT = 256