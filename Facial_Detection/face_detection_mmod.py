import cv2
import dlib

image_to_detect = cv2.imread("../Images/tyson_francis.jpg")

face_detection_classifier = dlib.cnn_face_detection_model_v1("../Models/mmod_human_face_detector.dat")

all_face_locations = face_detection_classifier(image_to_detect, 1)

for index, current_face_location in enumerate(all_face_locations):
    
    left_x, left_y, right_x, right_y = current_face_location.rect.left(), current_face_location.rect.top(), current_face_location.rect.right(), current_face_location.rect.bottom()

    print("Found face {} at left: {}, top: {}, right: {}, bottom: {}".format(index + 1, left_x, left_y, right_x, right_y))

    current_face_image = image_to_detect[left_y: right_y, left_x: right_x]

    cv2.imshow("Face No " + str(index + 1), current_face_image)

    cv2.rectangle(image_to_detect, (left_x, left_y), (right_x, right_y), (0, 0, 255), 2)

cv2.imshow("Faces to detect in an image", image_to_detect)

cv2.waitKey(0)

cv2.destroyAllWindows()