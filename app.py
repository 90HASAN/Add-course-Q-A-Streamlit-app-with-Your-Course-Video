import numpy as np
import pandas as pd
import streamlit as st
import torch
import joblib
from urllib.parse import urlparse, parse_qs
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
torch.set_num_threads(4)

st.set_page_config(page_title="Sigma Course Assistant", page_icon="🎓", layout="centered")

@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


@st.cache_resource(show_spinner="Loading course data...")
def load_data():
    return joblib.load("embaddings.joblib")


@st.cache_resource
def load_groq_client():
    return Groq() 


model = load_model()
df = load_data()
client = load_groq_client()


VIDEO_ID_MAP = {
    "1": "https://www.youtube.com/watch?v=tVzUXW6siu0&list=PLu0W_9lII9agq5TrH9XLIKQvv0iaF2X3w",
    "2": "https://www.youtube.com/watch?v=kJEsTjH5mVg&list=PLu0W_9lII9agq5TrH9XLIKQvv0iaF2X3w&index=2",
    "3": "https://www.youtube.com/watch?v=kJEsTjH5mVg&list=PLu0W_9lII9agq5TrH9XLIKQvv0iaF2X3w&index=2",
    "4": "https://www.youtube.com/watch?v=nXba2-mgn1k&list=PLu0W_9lII9agq5TrH9XLIKQvv0iaF2X3w&index=4",
    "5": "https://www.youtube.com/watch?v=1BsVhumGlNc&list=PLu0W_9lII9agq5TrH9XLIKQvv0iaF2X3w&index=5",
}


def extract_video_id(url_or_id: str):
    """Accepts a full YouTube URL (watch?v=, youtu.be/, with or without
    playlist/index params) or a bare video ID, and returns just the ID."""
    if not url_or_id:
        return None
    if "youtu" not in url_or_id:
        return url_or_id 

    parsed = urlparse(url_or_id)
    if "v" in parse_qs(parsed.query):
        return parse_qs(parsed.query)["v"][0]
    return parsed.path.lstrip("/").split("/")[0] or None


def youtube_link(video_number, start_seconds):
    raw = VIDEO_ID_MAP.get(str(video_number))
    video_id = extract_video_id(raw)
    if not video_id:
        return None
    return f"https://www.youtube.com/watch?v={video_id}&t={int(start_seconds)}s"


def get_top_chunks(question: str, top_k: int = 5) -> pd.DataFrame:
    question_embedding = model.encode([question])[0]
    similarities = cosine_similarity(
        np.vstack(df["embedding"]), [question_embedding]
    ).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return df.iloc[top_indices]


def build_prompt(question: str, top_chunks: pd.DataFrame) -> str:
    return f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{top_chunks[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{question}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course.

Important: only point the user to a video/timestamp if the chunk is genuinely relevant to what they asked. If none of the provided chunks actually answer the question (for example, the word overlap is just a coincidence, or a topic is only mentioned in passing without real explanation), clearly say that this specific topic isn't covered in the course rather than stretching an unrelated mention into an answer. It is better to honestly say something isn't covered than to force a weak connection.
'''


def ask_groq(prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def format_timestamp(seconds) -> str:
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"



st.title("🎓 Sigma Course Assistant")
st.caption("Ask anything about the course and get pointed to the exact video & timestamp.")

if "history" not in st.session_state:
    st.session_state.history = [] 

question = st.chat_input("Ask a question about the course...")

if question:
    with st.spinner("Searching course videos..."):
        top_chunks = get_top_chunks(question, top_k=5)
        prompt = build_prompt(question, top_chunks)
        answer = ask_groq(prompt)
    st.session_state.history.append((question, answer, top_chunks))

# Render conversation, most recent first
for q, a, chunks in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
        with st.expander("Matched video sections"):
            for _, row in chunks.iterrows():
                timestamp_str = f"{format_timestamp(row['start'])} → {format_timestamp(row['end'])}"
                link = youtube_link(row["number"], row["start"])
                if link:
                    st.markdown(
                        f"**{row['title']}** (Video {row['number']}) — "
                        f"[`{timestamp_str}`]({link})"
                    )
                else:
                    st.markdown(
                        f"**{row['title']}** (Video {row['number']}) — `{timestamp_str}`"
                    )

if not st.session_state.history:
    st.info("Type a question below to get started, e.g. *\"Where is useState explained?\"*")