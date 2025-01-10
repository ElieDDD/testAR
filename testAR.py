import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Funny Face Filter", layout="wide")
st.title("Funny Face Filter App")

# Load the funny filters (mustache and glasses)
def load_filter_images():
    mustache = cv2.imread("mustache.png", cv2.IMREAD_UNCHANGED)
    glasses = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)
    if mustache is None or glasses is None:
        st.error("Error: Filter images (mustache.png or glasses.png) are not loading. Ensure they are in the correct directory.")
    return mustache, glasses

mustache_img, glasses_img = load_filter_images()

def overlay_filter(frame, filter_img, x, y, w, h):
    # Resize the filter to fit the face feature
    filter_resized = cv2.resize(filter_img, (w, h))
    if filter_resized.shape[2] == 4:  # Ensure the filter has an alpha channel
        for c in range(0, 3):  # Loop through color channels
            frame[y:y+h, x:x+w, c] = frame[y:y+h, x:x+w, c] * (1 - filter_resized[:, :, 3] / 255) + filter_resized[:, :, c] * (filter_resized[:, :, 3] / 255)
    else:
        st.error("Filter image does not have an alpha channel (transparency). Check your images.")

# Load the Haar Cascade for face and facial features detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_and_draw_filters(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        st.warning("No faces detected. Try adjusting lighting or camera position.")
    for (x, y, w, h) in faces:
        # Debug: Draw rectangles around detected faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

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
    st.sidebar.write("Use the Streamlit camera input below to upload an image.")

    # Use Streamlit's camera input
    img_file = st.camera_input("Take a picture")

    if img_file is not None:
        # Convert the uploaded image to OpenCV format
        img = Image.open(img_file)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Apply funny filters
        frame_with_filters = detect_and_draw_filters(frame)

        # Display the resulting image
        st.image(frame_with_filters, channels="BGR", caption="With Funny Filters")

if __name__ == "__main__":
    main()
