import cv2
import dlib
import numpy as np

# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("../Models/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("../Models/dlib_face_recognition_resnet_model_v1.dat")

# Capture video from the default camera
webcam_video_stream = cv2.VideoCapture(0)

# Function to convert dlib rectangle to bounding box
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

# Function to convert dlib full object detection to face encoding
def face_encodings(image, face_rect):
    shape = sp(image, face_rect)
    return np.array(facerec.compute_face_descriptor(image, shape))

# Load and encode faces
def encode_face(image_path):
    image = dlib.load_rgb_image(image_path)
    detected_faces = detector(image, 1)
    if len(detected_faces) > 0:
        return face_encodings(image, detected_faces[0])
    return None

francis_face_encodings = encode_face("../Images/francis_face.jpg")
tyson_face_encodings = encode_face("../Images/tyson_face.jpg")
htoo_face_encodings = encode_face("../Images/htoo.jpg")

# Store the encodings and names
known_face_encodings = [francis_face_encodings, tyson_face_encodings, htoo_face_encodings]
known_face_names = ["Francis Ngannou", "Tyson Fury", "Htoo"]

process_this_frame = True

# Main loop for face recognition
while True:
    ret, frame = webcam_video_stream.read()
    if not ret:
        break

    # Resize frame for faster face detection processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        # Detect faces
        detected_faces = detector(rgb_small_frame, 1)
        face_locations = [rect_to_bb(face) for face in detected_faces]
        current_face_encodings = [face_encodings(rgb_small_frame, face) for face in detected_faces]

    process_this_frame = not process_this_frame

    for (x, y, w, h), face_encoding in zip(face_locations, current_face_encodings):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        x, y, w, h = x*4, y*4, w*4, h*4

        # Draw a rectangle around each face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Compare face with known faces
        name = "Unknown"
        if face_encoding is not None:
            for known_face_encoding, known_name in zip(known_face_encodings, known_face_names):
                if known_face_encoding is not None:
                    matches = np.linalg.norm(known_face_encoding - face_encoding) <= 0.6
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