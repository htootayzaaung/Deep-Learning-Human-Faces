import mediapipe as mp
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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
        # Convert the MediaPipe landmark data to a PIL image
        pil_image = Image.fromarray(current_frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        for face_landmarks in results.multi_face_landmarks:
            # Draw the points
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                draw.point((x, y), 'green')

            # Draw the mesh lines
            for connection in mp_face_mesh.FACEMESH_TESSELATION:
                start_idx = connection[0]
                end_idx = connection[1]
                start_point = face_landmarks.landmark[start_idx]
                end_point = face_landmarks.landmark[end_idx]
                start_point = (int(start_point.x * w), int(start_point.y * h))
                end_point = (int(end_point.x * w), int(end_point.y * h))
                draw.line([start_point, end_point], fill='green', width=1)

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
