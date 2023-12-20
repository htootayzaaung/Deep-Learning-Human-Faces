import cv2
import dlib

# Capture the video from the default webcam
webcam_video_stream = cv2.VideoCapture(0)

# Load the pretrained HOG SVN model for face detection
face_detection_classifier = dlib.get_frontal_face_detector()

# Load the shape predictor for face landmarks
face_shape_predictor = dlib.shape_predictor('../Models/shape_predictor_68_face_landmarks.dat')

# Loop through every frame in the video
while True:
    ret, current_frame = webcam_video_stream.read()
    if not ret:
        break

    # Convert BGR (OpenCV format) to RGB (Dlib format)
    current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

    # Detect all face locations using the HOG SVN classifier
    all_face_locations = face_detection_classifier(current_frame_rgb, 1)
    face_landmarks = dlib.full_object_detections()

    for current_face_location in all_face_locations:
        face_landmarks.append(face_shape_predictor(current_frame_rgb, current_face_location))

    # Check if any face landmarks were found before proceeding
    if len(face_landmarks) > 0:
        all_face_chips = dlib.get_face_chips(current_frame_rgb, face_landmarks)

        for index, current_face_chip in enumerate(all_face_chips):
            # Convert RGB (Dlib format) back to BGR (OpenCV format) for display
            current_face_chip_bgr = cv2.cvtColor(current_face_chip, cv2.COLOR_RGB2BGR)
            cv2.imshow("Face no " + str(index + 1), current_face_chip_bgr)

    cv2.imshow("Webcam Video", current_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_video_stream.release()
cv2.destroyAllWindows()
