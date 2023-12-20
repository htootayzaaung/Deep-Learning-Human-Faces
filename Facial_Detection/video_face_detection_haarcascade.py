import cv2

# Capture the video from the default camera
webcam_video_stream = cv2.VideoCapture("../Videos/Compressed/Bella_Ciao.mp4")

face_detection_classifier = cv2.CascadeClassifier("../Models/haarcascades/haarcascade_frontalface_default.xml")

# Initialize the array varaible to hold all the face locations in the frame
all_face_locations = []

# loop through every frame in the video
while True:
    # get the current frame from the video stream as an image
    ret, current_frame = webcam_video_stream.read()

    # resize the current frame to 1/4 size to process faster
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)

    # detect all faces in the image
    all_face_locations = face_detection_classifier.detectMultiScale(current_frame_small)
    
    for index, current_face_location in enumerate (all_face_locations):

        x, y, width, height = current_face_location

        left_pos = x
        top_pos = y
        right_pos = x + width
        bottom_pos = y + height

        print("Found face {} at top: {}, right: {}, bottom: {}, left: {}".format(index + 1, top_pos, right_pos, bottom_pos, left_pos))

        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)

    cv2.imshow("Web-Cam Video", current_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
webcam_video_stream.release()
cv2.destroyAllWindows()

