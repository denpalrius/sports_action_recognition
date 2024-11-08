import streamlit as st
import moviepy.editor as mp
import numpy as np
import cv2
import keras
from keras.models import load_model

import os

os.makedirs("temp", exist_ok=True)

# Loading the CNN-RNN model
MODEL_PATH = '../../models/cnn_rnn_ucf101_10c_tl_v1.keras'
model = load_model(MODEL_PATH)

print("Model loaded successfully!")

# Constants
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
sports_actions = [
    "SkyDiving", "Biking", "HorseRace", "Surfing", "TennisSwing",
    "Punch", "Basketball", "JumpRope", "Archery", "Skiing"
]

# Feature extraction with InceptionV3
def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet", include_top=False, pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3))
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

# Preprocessing functions
def crop_center_square(frame):
    y, x = frame.shape[:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

def load_and_preprocess_video(video_path, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = crop_center_square(frame)
        frame = cv2.resize(frame, resize)
        frame = frame[..., ::-1]  # Convert BGR to RGB
        frames.append(frame)
    cap.release()
    return np.array(frames)

def create_video_mask_and_features(frames):
    frames = frames[None, ...]
    video_length = min(MAX_SEQ_LENGTH, frames.shape[1])

    mask = np.zeros((1, MAX_SEQ_LENGTH), dtype="bool")
    features = np.zeros((1, MAX_SEQ_LENGTH, 2048), dtype="float32")

    for j in range(video_length):
        features[0, j, :] = feature_extractor.predict(frames[:, j, :], verbose=0)
    mask[0, :video_length] = 1
    return features, mask

def convert_to_mp4(video_path):
    mp4_path = video_path.replace(".avi", ".mp4")
    clip = mp.VideoFileClip(video_path)
    clip.write_videofile(mp4_path, codec="libx264")
    return mp4_path

st.title("Sports Action Recognition")
st.write("Upload a video file to predict the sports action.")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

K = 5

if uploaded_file is not None:
    video_path = os.path.join("temp", uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Convert to .mp4 if the file is in .avi format
    if video_path.endswith(".avi"):
        video_path = convert_to_mp4(video_path)
    
    st.video(video_path)

    # Preprocess video
    frames = load_and_preprocess_video(video_path)
    if frames.shape[0] < MAX_SEQ_LENGTH:
        st.warning(f"Video is too short. Please upload a video with at least {MAX_SEQ_LENGTH} frames.")
    else:
        frame_features, frame_mask = create_video_mask_and_features(frames)

        # Make prediction
        preds = model.predict([frame_features, frame_mask])
        predicted_label = sports_actions[np.argmax(preds)]
        confidence = np.max(preds)

        # Display prediction
        st.write(f"**Predicted Action:** {predicted_label}")
        st.write(f"**Confidence:** {confidence:.2f}")
        
        # Get top K predictions
        top_k_indices = np.argsort(preds[0])[::-1][:K]
        top_k_actions = [sports_actions[i] for i in top_k_indices]
        top_k_confidences = preds[0][top_k_indices]

        # Display top K predictions
        st.write("**Top Predictions:**")
        for i in range(K):
            st.write(f"{i+1}. **{top_k_actions[i]}** with confidence **{top_k_confidences[i]:.2f}**")
else:
    st.write("Please upload a video to test the model.")