import cv2

# Load the cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture the video from the default camera
webcam_video_stream = cv2.VideoCapture(0)

# Loop through every frame in the video
while True:
    # Get the current frame from the video stream as an image
    ret, current_frame = webcam_video_stream.read()
    if not ret:
        break

    # Convert the current frame to grayscale
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    all_face_locations = face_cascade.detectMultiScale(current_frame_gray, scaleFactor=1.1, minNeighbors=5)

    # Loop through all the faces found
    for index, face_location in enumerate(all_face_locations):
        left_pos, top_pos, width, height = face_location

        print("Found face {} at top: {}, right: {}, bottom: {}, left: {}".format(index + 1, top_pos, left_pos + width, top_pos + height, left_pos))

        # Draw rectangle around the face detected
        cv2.rectangle(current_frame, (left_pos, top_pos), (left_pos + width, top_pos + height), (0, 0, 255), 2)

    # Display the current frame with boxes around detected faces
    cv2.imshow("Web-Cam Video", current_frame)

    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
webcam_video_stream.release()
cv2.destroyAllWindows()
