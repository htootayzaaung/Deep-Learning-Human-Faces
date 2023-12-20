import mediapipe as mp
import cv2
from PIL import Image, ImageDraw
import numpy as np

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Define drawing specifications for the mesh
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

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
            
            # Draw the facial landmarks and mesh edges on the frame
            mp.solutions.drawing_utils.draw_landmarks(
                image=current_frame_rgb,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            # Convert the PIL image back to an array and replace the current frame
            current_frame = np.array(pil_image)

    # Display the frame with landmarks.
    cv2.imshow("Webcam Facial Landmarks", current_frame)

    # Press 'q' to quit the video stream.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows.
webcam_video_stream.release()
cv2.destroyAllWindows()
