"""
EchoVision Unified Integrator
Main orchestration combining Reflex Loop (high priority) and Cognitive Loop (low priority).
"""
import sys
import threading
import time
import datetime
import os
import cv2

import config
from navigation_system import NavigationSystem
from gui import NavigationGUI

# Cognitive Loop Imports
sys.path.append(os.path.join(os.path.dirname(__file__), "Video Chatbot"))
# Ensure imports for ingest, chat, Image_Captioning work
from video_chatbot.src import ingest
from video_chatbot.src.chat import ChatBot
from video_captioning_src.image_captioning.Image_Captioning import ImageCaptioning
from video_captioning_src.saving_caption_to_file.save_caption_to_file import save_captions


class SharedFrameBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.timestamp = None

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame.copy() if frame is not None else None
            self.timestamp = datetime.datetime.now()

    def get_latest_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None


class UnifiedOrchestrator:
    def __init__(self):
        self.frame_buffer = SharedFrameBuffer()
        
        # Initialize Reflex Subsystem
        self.nav_system = NavigationSystem()
        
        # Intercept frame processing to copy to our thread-safe buffer
        self._original_process_frame = self.nav_system.process_frame
        self.nav_system.process_frame = self._intercept_frame
        
        # Initialize GUI (Runs the Reflex Loop on the Main Thread)
        self.gui = NavigationGUI(self.nav_system)
        
        # Initialize Cognitive AI Components (Lazy load to prevent blocking)
        self.image_captioner = None
        self.chat_bot = None
        
        self.running = False

    def _intercept_frame(self):
        """Intercepts Reflex Loop frames to share with Cognitive Loop"""
        result = self._original_process_frame()
        if result and result.get('frame') is not None:
            self.frame_buffer.update_frame(result['frame'])
        return result

    def start_cognitive_workflow(self):
        """Sequential Workflow Thread"""
        print("\n=== COGNITIVE PIPELINE INITIALIZING ===")
        
        # Initialize heavy models once
        print("[System] Loading VLM (Qwen)...")
        self.image_captioner = ImageCaptioning()
        
        docs_dir = os.path.join("Video Chatbot", "video_chatbot", "docs_dir")
        os.makedirs(docs_dir, exist_ok=True)
        analysis_filepath = os.path.join(docs_dir, "video_analysis.txt")
        
        while self.running:
            # Wait for camera to be active and get a valid frame
            if not self.nav_system.is_running or self.frame_buffer.get_latest_frame() is None:
                time.sleep(1)
                continue
                
            frame = self.frame_buffer.get_latest_frame()
            print("\n[➔] Frame Captured from Tkinter.")
            
            try:
                print("[➔] Sending to VLM to generate caption...")
                caption = self.image_captioner.generate_caption_from_frame(frame)
                print(f"    CAPTION: {caption}")
                
                print("[➔] Saving Caption...")
                with open(analysis_filepath, "a", encoding="utf-8") as f:
                    f.write(caption + "\n")
                
                print("[➔] Indexing to RAG Pipeline (ChromaDB)...")
                ingest.create_vector_db()
                
                print("[➔] Initializing LLM and Asking Question...")
                # Re-init ChatBot so it catches the newest ChromaDB updates context
                self.chat_bot = ChatBot()
                    
                question = "Based on the recent video analysis, explain what is happening in the current view."
                response = self.chat_bot.ask(question)
                
                print(f"\n=== FINAL LLM ANSWER ===")
                print(response)
                print("========================\n")
                
                if self.nav_system.audio_feedback:
                    self.nav_system.audio_feedback.speak(str(response), priority=False)
                    
            except Exception as e:
                print(f"Workflow Error: {e}")
                
            # Wait before initiating the next automated sequence
            time.sleep(5)

    def run(self):
        self.running = True

        workflow_thread = threading.Thread(target=self.start_cognitive_workflow, daemon=True)
        workflow_thread.start()

        # Tkinter GUI (Reflex Loop window) must run on the Main Thread for Windows UI safety
        try:
            print("[Thread 1] Starting Reflex GUI on Main Thread...")
            # Auto-start detection immediately so the camera captures 
            self.gui.root.after(1500, self.gui.toggle_detection)
            self.gui.run()
        except KeyboardInterrupt:
            print("\nShutting down EchoVision Unified System...")
        finally:
            self.running = False
            self.nav_system.cleanup()
            print("Shutdown complete.")

if __name__ == "__main__":
    orchestrator = UnifiedOrchestrator()
    orchestrator.run()
