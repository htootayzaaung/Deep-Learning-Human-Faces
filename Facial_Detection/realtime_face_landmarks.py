import face_recognition
import cv2
from PIL import Image, ImageDraw
import numpy as np

# Capture the video from the default camera
webcam_video_stream = cv2.VideoCapture(0)

# loop through every frame in the video
while True:
    # Get the current frame from the video stream as an image
    ret, current_frame = webcam_video_stream.read()
    # Resize the current frame to 1/4 size to process faster
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)
    # Find all the face locations and face landmarks in the current frame
    # Note that if you're not using the small image, remove the '_small' suffix below
    face_landmarks_list = face_recognition.face_landmarks(current_frame_small)
    # Print the number of faces detected
    print(f"Number of faces detected: {len(face_landmarks_list)}")

    # Loop over each detected face in the frame
    for face_landmarks in face_landmarks_list:
        # Convert the detected face landmarks (which are in 1/4 scale) back to full size
        face_landmarks = {key: [(int(point[0] * 4), int(point[1] * 4)) for point in value] for key, value in face_landmarks.items()}
        # Draw the face landmarks on the frame
        pil_image = Image.fromarray(current_frame)
        d = ImageDraw.Draw(pil_image)
        for facial_feature in face_landmarks.keys():
            d.line(face_landmarks[facial_feature], width=5, fill='green')
            for point in face_landmarks[facial_feature]:
                d.ellipse([(point[0] - 2, point[1] - 2), (point[0] + 2, point[1] + 2)], fill='red')

        # Convert PIL image back to array and show the frame
        current_frame = np.array(pil_image)
        cv2.imshow("Webcam Video", current_frame)

    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
webcam_video_stream.release()
cv2.destroyAllWindows()
