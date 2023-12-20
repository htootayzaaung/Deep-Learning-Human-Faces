import face_recognition
import cv2

# Capture the video from the default camera
webcam_video_stream = cv2.VideoCapture(0)

# Initialize the array variable to hold all the face locations in the frame
all_face_locations = []

# loop through every frame in the video
while True:
    # get the current frame from the video stream as an image
    ret, current_frame = webcam_video_stream.read()

    # resize the current frame to 1/4 size to process faster
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)

    # detect all faces in the image
    all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=2, model="hog")

    for index, current_face_location in enumerate(all_face_locations):
        top_pos, right_pos, bottom_pos, left_pos = current_face_location

        top_pos = top_pos * 4
        right_pos = right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4

        print("Found face {} at top: {}, right: {}, bottom: {}, left: {}".format(index + 1, top_pos, right_pos, bottom_pos, left_pos))

        # Slicing the current face from the main frame
        current_face_image = current_frame[top_pos:bottom_pos, left_pos:right_pos]

        # Blur the sliced face
        current_face_image = cv2.GaussianBlur(current_face_image, (99, 99), 30)

        # Replace the original face region with the blurred version
        current_frame[top_pos:bottom_pos, left_pos:right_pos] = current_face_image

        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)

    cv2.imshow("Web-Cam Video", current_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
webcam_video_stream.release()
cv2.destroyAllWindows()
