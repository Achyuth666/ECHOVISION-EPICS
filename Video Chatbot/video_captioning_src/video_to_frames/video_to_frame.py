import cv2
import os


def extract_frames_from_video(video_path, output_folder, interval=1):
    """
    Extracting the frames from video every 'interval' seconds.
    """
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found at: {video_path}")
        return 0

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    video_to_capture = cv2.VideoCapture(video_path)

    if not video_to_capture.isOpened():
        print("ERROR: Could not open video. Check filepath.")
        return 0

    fps = video_to_capture.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        print("ERROR: FPS is 0 - Video might be corrupted.")
        return 0

    fps = int(fps)
    process_every_n_frames = fps * interval
    frame_count = 0
    saved_count = 0

    print(f"Extracting frames from: {video_path}")

    while video_to_capture.isOpened():
        success, frame = video_to_capture.read()
        if not success:
            break

        if frame_count % process_every_n_frames == 0:
            # Calculating the timestamp
            seconds = frame_count // fps

            # Saving the frame
            filename = f"frame_{seconds:04d}.jpg"
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, frame)
            saved_count += 1

        frame_count += 1

    video_to_capture.release()
    print(f"Done! Extracted {saved_count} frames to {output_folder}")
    return saved_count


# try run
# if __name__ == "__main__":
#     video_file = r"C:\Users\Achyu\OneDrive\Desktop\ElevateTrust AI\Video_ChatBot\test_inputs\input2.mp4"
#     output_dir = r"C:\Users\Achyu\OneDrive\Desktop\ElevateTrust AI\Video_ChatBot\frames"
#     extract_frames_from_video(video_file, output_dir, interval=1)