import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import cv2


class ImageCaptioning:
    def __init__(self):

        print("Loading Qwen2-VL Vision Model...")
        self.model_path = "Qwen/Qwen2-VL-2B-Instruct"

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def generate_caption_from_frame(self, cv2_frame):
        """
        Adapts Image_Captioning to accept a cv2 frame array in memory rather than a saved image file path.
        """
        import datetime
        from PIL import Image

        rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        timestamp = datetime.datetime.now().strftime("%M:%S")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": "You are a vigilant observer tasked with documenting exactly what is happening in the current frame of video footage. Describe the scene as precisely and objectively as possible, focusing on visible actions, objects, people, positions, and interactions. Be concise and direct, but do not omit any details — even minor or seemingly irrelevant ones may be important later. Do not summarize across time or interpret intent. Just report what is visible in this single frame."}
            ]
        }]

        # Preprocess
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=100)

        # Decode
        output_text = self.processor.batch_decode(
            [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)],
            skip_special_tokens=True
        )[0]

        log_entry = f"[{timestamp}] {output_text}"
        return log_entry

    def generate_captions(self, frames_folder):
        """
        Reads images from frames_folder and returns a list of captions.
        """
        print(f"Processing frames from: {frames_folder}")
        captions = []


        files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])

        if not files:
            print("No jpg files found in frames folder!")
            return []

        for file in files:
            image_path = os.path.join(frames_folder, file)

            try:
                seconds = int(file.split('_')[1].split('.')[0])
                timestamp = f"{seconds // 60:02}:{seconds % 60:02}"
            except:
                timestamp = "00:00"

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "You are a vigilant observer tasked with documenting exactly what is happening in the current frame of video footage. Describe the scene as precisely and objectively as possible, focusing on visible actions, objects, people, positions, and interactions. Be concise and direct, but do not omit any details — even minor or seemingly irrelevant ones may be important later. Do not summarize across time or interpret intent. Just report what is visible in this single frame."}
                ]
            }]

            # Preprocess
            text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )

            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=100)

            # Decode
            output_text = self.processor.batch_decode(
                [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)],
                skip_special_tokens=True
            )[0]

            log_entry = f"[{timestamp}] {output_text}"
            print(log_entry)
            captions.append(log_entry)

        return captions


