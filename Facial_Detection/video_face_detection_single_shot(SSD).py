import cv2
import numpy as np

# Capture the video from the default webcam
webcam_video_stream = cv2.VideoCapture(0)  # Changed to '0' for default webcam

face_detection_classifier = cv2.dnn.readNetFromCaffe('../Models/deploy.prototxt', '../Models/res10_300x300_ssd_iter_140000.caffemodel')

# Loop through every frame in the video
while True:
    # Get the current frame from the video stream as an image
    ret, current_frame = webcam_video_stream.read()
    if not ret:
        break

    img_height, img_width = current_frame.shape[:2]

    resized_image = cv2.resize(current_frame, (300, 300))

    image_to_detect_blob = cv2.dnn.blobFromImage(resized_image, 1.0, (300, 300), (104, 177, 123))

    face_detection_classifier.setInput(image_to_detect_blob)

    all_face_locations = face_detection_classifier.forward()

    no_of_detections = all_face_locations.shape[2]

    for index in range(no_of_detections):
        detection_confidence = all_face_locations[0, 0, index, 2]
        
        if detection_confidence > 0.5:
            current_face_location = all_face_locations[0, 0, index, 3:7] * np.array([img_height, img_width, img_width, img_height])
            left_x, left_y, right_x, right_y = current_face_location.astype("int")

            print('Found face {} at left_x:{}, left_y:{}, right_x:{}, right_y:{}'.format(index+1, left_x, left_y, right_x, right_y))
            cv2.rectangle(current_frame, (left_x, left_y), (right_x, right_y), (0, 255, 0), 2)

    cv2.imshow("Web-Cam Video", current_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
webcam_video_stream.release()
cv2.destroyAllWindows()
