import mediapipe as mp
import cv2
import numpy as np
from PIL import Image

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Read an image from your directory.
image_path = '../Images/htoo.jpg'
image = cv2.imread(image_path)
h, w, _ = image.shape

# Convert the color space from BGR (OpenCV) to RGB.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and get landmarks.
results = face_mesh.process(image_rgb)

# Overlay landmarks on the RGB image.
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for landmark in face_landmarks.landmark:
            # Scale landmark coordinates to image dimensions
            x = min(int(landmark.x * w), w - 1)
            y = min(int(landmark.y * h), h - 1)
            # Draw a small circle on the landmark coordinates
            cv2.circle(image_rgb, (x, y), 8, (0, 255, 0), -1)

# Convert the RGB image (with landmarks) back to a PIL Image object.
pil_image_with_landmarks = Image.fromarray(image_rgb)

# Display the image using PIL.
pil_image_with_landmarks.show()

# Save the image with landmarks using PIL.
pil_image_with_landmarks.save('../Images/htoo_with_landmarks.jpg')

# Release resources.
face_mesh.close()
