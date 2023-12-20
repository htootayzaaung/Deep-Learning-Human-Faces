import mediapipe as mp
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Start capturing video from the default camera.
webcam_video_stream = cv2.VideoCapture(0)

# Loop through every frame in the video stream.
while True:
    # Get the current frame from the video stream.
    ret, current_frame = webcam_video_stream.read()
    if not ret:
        continue

    # Convert the color space from BGR (OpenCV) to RGB.
    current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    h, w, _ = current_frame.shape

    # Process the frame and get landmarks.
    results = face_mesh.process(current_frame_rgb)

    # Overlay landmarks on the RGB frame.
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Convert the MediaPipe landmark data to a PIL image
            pil_image = Image.fromarray(current_frame_rgb)
            draw = ImageDraw.Draw(pil_image)

            # Define a smaller radius for the dots
            dot_radius = 1.5  # Decrease this value to make the dots smaller

            for landmark in face_landmarks.landmark:
                # Scale landmark coordinates to frame dimensions
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                # Draw a smaller circle on the landmark coordinates
                draw.ellipse([(x - dot_radius, y - dot_radius),
                              (x + dot_radius, y + dot_radius)], fill=(0, 255, 0))

            # Convert the PIL image back to an RGB array
            current_frame_rgb_array = np.array(pil_image)
            # Convert the RGB array back to BGR, which OpenCV expects
            current_frame = cv2.cvtColor(current_frame_rgb_array, cv2.COLOR_RGB2BGR)
    else:
        # If no landmarks are detected, just display the original frame
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

    # Display the frame with landmarks.
    cv2.imshow("Webcam Facial Landmarks", current_frame)

    # Press 'q' to quit the video stream.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows.
webcam_video_stream.release()
cv2.destroyAllWindows()
