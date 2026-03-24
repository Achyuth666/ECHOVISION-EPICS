import os
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from .config import MODEL_PATH, CONTEXT_WINDOW, MAX_OUTPUT


def qwen_messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == MessageRole.SYSTEM:
            prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
        elif message.role == MessageRole.USER:
            prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
        elif message.role == MessageRole.ASSISTANT:
            prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"

    if not prompt.endswith("<|im_start|>assistant\n"):
        prompt += "<|im_start|>assistant\n"
    return prompt


def setup_ai_models():
    print(f"Loading Local Qwen Model.")

    import multiprocessing
    optimal_threads = max(1, multiprocessing.cpu_count() - 2)

    llm = LlamaCPP(
        model_path=MODEL_PATH,
        temperature=0.1,
        max_new_tokens=128,
        context_window=2048,
        messages_to_prompt=qwen_messages_to_prompt,
        generate_kwargs={
            "stop": ["<|im_end|>", "<|endoftext|>"],
        },
        model_kwargs={
            "n_gpu_layers": -1,
            "n_threads": optimal_threads,
            "n_batch": 512,
            "repeat_penalty": 1.2,
        },
        verbose=False
    )

    print("Loading Embeddings...")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    Settings.llm = llm
    Settings.embed_model = embed_model

    Settings.system_prompt = (
        "You are a video analysis assistant. "
        "Answer clearly and completely based only on the video content. "
        "Use as many sentences as needed to fully answer the question."
    )

    print(f"AI Ready.")
    return llm, embed_model