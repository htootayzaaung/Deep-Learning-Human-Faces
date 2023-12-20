import cv2
import dlib

# Capture the video from the default camera
webcam_video_stream = cv2.VideoCapture(0)  # Changed from file path to '0' for default webcam

# Load the pretrained HOG SVN model
face_detection_classifier = dlib.get_frontal_face_detector()

# Loop through every frame in the video
while True:
    # Get the current frame from the video stream as an image
    ret, current_frame = webcam_video_stream.read()
    # Check if frame is grabbed
    if not ret:
        print("Failed to grab frame")
        break

    # Create a grayscale image to pass into the dlib HOG detector
    current_frame_to_detect_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    # Resize the current frame to 1/4 size to process faster
    current_frame_small = cv2.resize(current_frame_to_detect_gray, (0, 0), fx=0.25, fy=0.25)
    # Detect all face locations using the HOG SVN classifier
    all_face_locations = face_detection_classifier(current_frame_small, 1)

    # Looping through the face locations
    for index, current_face_location in enumerate(all_face_locations):
        # Start and end coordinates
        left_pos, top_pos, right_pos, bottom_pos = (current_face_location.left(), current_face_location.top(),
                                                    current_face_location.right(), current_face_location.bottom())

        # Change the position magnitude to fit the actual size video frame
        left_pos *= 4
        top_pos *= 4
        right_pos *= 4
        bottom_pos *= 4

        # Printing the location of current face
        print('Found face {} at top:{}, right:{}, bottom:{}, left:{}'.format(index + 1, top_pos, right_pos, bottom_pos, left_pos))
        # Draw rectangle around the face detected
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)

    # Showing the current face with rectangle drawn
    cv2.imshow("Webcam Video", current_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the stream and cam
# Close all OpenCV windows open
webcam_video_stream.release()
cv2.destroyAllWindows()
