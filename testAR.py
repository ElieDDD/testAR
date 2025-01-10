import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Funny Face Filter", layout="wide")
st.title("Funny Face Filter App")
cap = cv2.VideoCapture(0)  # Try 0, 1, or 2 depending on your setup
if not cap.isOpened():
    print("Error: Unable to access the camera")
# Load the funny filters (mustache and glasses)
def load_filter_images():
    mustache = cv2.imread("mustache.png", cv2.IMREAD_UNCHANGED)
    glasses = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)
    return mustache, glasses

mustache_img, glasses_img = load_filter_images()

def overlay_filter(frame, filter_img, x, y, w, h):
    # Resize the filter to fit the face feature
    filter_resized = cv2.resize(filter_img, (w, h))
    for c in range(0, 3):  # Loop through color channels
        frame[y:y+h, x:x+w, c] = frame[y:y+h, x:x+w, c] * (1 - filter_resized[:, :, 3] / 255) + filter_resized[:, :, c] * (filter_resized[:, :, 3] / 255)

# Load the Haar Cascade for face and facial features detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_and_draw_filters(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        # Draw glasses on the face
        glasses_w = int(w * 0.8)
        glasses_h = int(h * 0.2)
        glasses_x = x + int(w * 0.1)
        glasses_y = y + int(h * 0.2)
        overlay_filter(frame, glasses_img, glasses_x, glasses_y, glasses_w, glasses_h)

        # Draw mustache under the nose
        mustache_w = int(w * 0.6)
        mustache_h = int(h * 0.15)
        mustache_x = x + int(w * 0.2)
        mustache_y = y + int(h * 0.6)
        overlay_filter(frame, mustache_img, mustache_x, mustache_y, mustache_w, mustache_h)

    return frame

def main():
    st.sidebar.title("Controls")
    run = st.sidebar.checkbox("Run", value=True)

    # Start the webcam stream
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = detect_and_draw_filters(frame)

        stframe.image(frame, channels="RGB", use_column_width=True)

    cap.release()

if __name__ == "__main__":
    main()
