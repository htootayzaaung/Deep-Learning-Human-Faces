import cv2
import dlib
import numpy as np

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("../Models/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("../Models/dlib_face_recognition_resnet_model_v1.dat")

# Function to convert dlib's rectangle to a bounding box
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

# Function for face encoding
def face_encodings(image, face_rect):
    shape = sp(image, face_rect)
    return np.array(facerec.compute_face_descriptor(image, shape))

# Function to load and encode faces
def encode_face(image_path):
    image = dlib.load_rgb_image(image_path)
    detected_faces = detector(image, 1)
    if len(detected_faces) > 0:
        return face_encodings(image, detected_faces[0])
    return None

# Load and encode faces of Francis Ngannou, Tyson Fury, and Htoo
francis_face_encodings = encode_face("../Images/francis_face.jpg")
tyson_face_encodings = encode_face("../Images/tyson_face.jpg")
htoo_face_encodings = encode_face("../Images/htoo.jpg")

# Store the encodings and names
known_face_encodings = [francis_face_encodings, tyson_face_encodings, htoo_face_encodings]
known_face_names = ["Francis Ngannou", "Tyson Fury", "Htoo"]

# Capture the video from a file
webcam_video_stream = cv2.VideoCapture('../Videos/tyson_fury.mp4')

# Main loop for face recognition
while True:
    ret, frame = webcam_video_stream.read()
    if not ret:
        webcam_video_stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Convert the image from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    detected_faces = detector(rgb_frame, 1)

    for face_rect in detected_faces:
        x, y, w, h = rect_to_bb(face_rect)

        # Draw a rectangle around each face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Encode the face found
        encoding = face_encodings(rgb_frame, face_rect)

        # Compare face with known faces
        name = "Unknown"
        if encoding is not None:
            for known_face_encoding, known_name in zip(known_face_encodings, known_face_names):
                if known_face_encoding is not None:
                    matches = np.linalg.norm(known_face_encoding - encoding) <= 0.6
                    if matches:
                        name = known_name
                        break

        # Display the name
        cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
webcam_video_stream.release()
cv2.destroyAllWindows()
