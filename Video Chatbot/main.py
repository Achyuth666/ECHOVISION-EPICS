import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
video_input = os.path.join(root_dir, "videos", "input.mp4")
frames_output = os.path.join(root_dir, "frames")
transcript_dir = os.path.join(root_dir, "video_chatbot", "docs_dir")

sys.path.append(os.path.join(root_dir, "video_captioning_src"))
sys.path.append(os.path.join(root_dir, "video_chatbot"))

from video_captioning_src.video_to_frames.video_to_frame import extract_frames_from_video
from video_captioning_src.image_captioning.Image_Captioning import ImageCaptioning
from video_chatbot.src.ingest import create_vector_db
from video_chatbot.src.chat import ChatBot


def main():
    print("VIDEO RAG PIPELINE")

    while True:
        print("\nOPTIONS:")
        print("1. Full Process (Extract, Caption, Ingest)")
        print("2. Start Chatting")
        print("3. Re-Sync Database (Manual Ingest)")
        print("4. Exit")

        choice = input("Select (1-4): ")

        if choice == "1":
            print("Extracting frames...")
            extract_frames_from_video(video_input, frames_output, interval=1)

            print("Generating captions...")
            captioner = ImageCaptioning()
            captions_list = captioner.generate_captions(frames_output)

            print("Saving transcript...")
            if not os.path.exists(transcript_dir):
                os.makedirs(transcript_dir)

            save_path = os.path.join(transcript_dir, "video_analysis.txt")
            with open(save_path, "w", encoding="utf-8") as f:
                for line in captions_list:
                    f.write(line + "\n")

            print("Updating database...")
            create_vector_db()
            print("Done.")

        elif choice == "2":
            try:
                bot = ChatBot()
                print("Chat Ready. Type 'exit' to quit.")
                while True:
                    q = input("\nYou: ")
                    if q.lower() in ["exit", "quit"]:
                        break
                    response = bot.ask(q)
                    print(f"Bot: {response}")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == "3":
            print("Reading docs_dir and updating database...")
            txt_file = os.path.join(transcript_dir, "video_analysis.txt")
            if not os.path.exists(txt_file):
                print(f"Error: No file found at {txt_file}")
                continue

            create_vector_db()
            print("Database updated.")

        elif choice == "4":
            break


if __name__ == "__main__":
    main()