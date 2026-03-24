import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="🎥 Video RAG Chatbot", layout="wide")
st.title("🎥 Video RAG Chatbot")

# ---- Video Upload ----
st.header("Upload Video")
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file:
    st.video(video_file)

    if st.button("Process Video"):
        with st.spinner("Processing video..."):
            response = requests.post(
                f"{BACKEND_URL}/upload",
                files={"video": video_file}
            )

        if response.status_code == 200:
            st.success("Video processed! You can now chat.")
            st.session_state["ready"] = True
        else:
            st.error(response.json().get("error"))

# ---- Chat Section ----
if st.session_state.get("ready"):
    st.header("💬 Chat with Video")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    user_input = st.text_input("Ask something about the video")

    if st.button("Ask") and user_input:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"question": user_input}
        )

        answer = response.json()["answer"]

        st.session_state.chat.append(("You", user_input))
        st.session_state.chat.append(("Bot", answer))

    for speaker, msg in st.session_state.chat:
        st.markdown(f"**{speaker}:** {msg}")