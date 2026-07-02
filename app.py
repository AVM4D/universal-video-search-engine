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
    os.makedirs(folder, exist_ok=True)

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

        all_frames=sorted(os.listdir(FRAMES_DIR))
        total_frames=len(all_frames)
        progress_bar.progress(10,text="AI is reading visual meanings... (0%)")
        for idx, filename in enumerate(all_frames):
            img_path =  os.path.join(FRAMES_DIR, filename)
            img = Image.open(img_path)
            
            with torch.no_grad():
                inputs=processor(images=img, return_tensors="pt").to(device)
                image_features = model.get_image_features(**inputs)
                embedding = image_features.pooler_output.cpu().numpy()[0].tolist()
                local_db[filename]=embedding
                pct=int((idx+1)/total_frames*100)
                progress_bar.progress(10+int(pct*0.9),text=f"AI is reading visual meanings... ({pct}%)")
        
        with open(db_path,"w") as f:
            json.dump(local_db,f)

        st.success("Video indexing complete! AI semantic vector database saved to disk")

    search_query = st.text_input("What are you looking for inside the video?",placeholder="Type a concept, action, or object...")

    if search_query:
        with open(db_path,"r") as f:
            video_db=json.load(f)
        st.write(f"Searching for: **{search_query}**")
        
        with torch.no_grad():
            text_inputs = processor(text=[search_query], return_tensors="pt", padding=True).to(device)
            text_features = model.get_text_features(**text_inputs)

            query_vector = text_features.pooler_output.cpu().numpy()[0]            
        
        import numpy as np

        results = []

        for filename, embedding in video_db.items():
            frame_vector = np.array(embedding)
            score=np.dot(query_vector,frame_vector)/(np.linalg.norm(query_vector)*np.linalg.norm(frame_vector))
            results.append((score,filename))
        
        results.sort(reverse=True)
        best_score, best_frame = results[0]

        matched_seconds = int(best_frame.split("_")[1].split(".")[0])

        st.subheader(f"Best Match Found at Second **{matched_seconds}** (Confidence: {best_score:.2f})")
        st.image(os.path.join(FRAMES_DIR, best_frame), use_container_width=True)

        








