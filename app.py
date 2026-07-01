import streamlit as st
import json 
import torch
import os
import cv2
import shutil
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

st.set_page_config(page_title="Universal Video Search Engine", page_icon="🔍", layout="wide")
st.title("🔍 Universal Semantic Video Search Engine")
st.write("Upload any video clip, let the AI index its visual meaning, and search through it instantly")

@st.cache_resource
def load_ai_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor, device

model, processor, device = load_ai_model()

UPLOAD_DIR = "user_uploads"
FRAMES_DIR = "user_frames"

for folder in [UPLOAD_DIR, FRAMES_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

uploaded_file = st.file_uploader("Choose a video file (.mp4, .mkv, .avi)", type=["mp4", "mkv", "avi"])

if uploaded_file is not None:
    video_path = os.path.join(UPLOAD_DIR,uploaded_file.name)
    with open(video_path,"wb") as f:
        f.write(uploaded_file.read())
    st.success(f"Successfully saved to disk: {video_path}")
    db_path = os.path.join(UPLOAD_DIR, f"{uploaded_file.name}_embeddings.json")
    if not os.path.exists(db_path):
        st.warning("New video detected. Initializing AI indexing pipeline ...")
        shutil.rmtree(FRAMES_DIR)
        os.makedirs(FRAMES_DIR)

        cap=cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        local_db = {}
        progress_bar = st.progress(0, text="Slicing video into seconds ...")

        frame_count=0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % fps == 0:
                seconds = frame_count // fps
                frame_name = f"frame_{seconds:04d}.jpg"
                img_path = os.path.join(FRAMES_DIR, frame_name)
                cv2.imwrite(img_path,frame)
            frame_count+=1
        cap.release()

        








