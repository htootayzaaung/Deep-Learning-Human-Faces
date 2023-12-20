import cv2
import dlib

image_to_detect = cv2.imread("Images/tyson_francis.jpg")

image_to_detect_gray = cv2.cvtColor(image_to_detect, cv2.COLOR_BGR2GRAY)

face_detection_classifier = dlib.get_frontal_face_detector()

all_face_locations = face_detection_classifier(image_to_detect, 1)

for index, current_face_location in enumerate(all_face_locations):
    
    left_x, left_y, right_x, right_y = current_face_location.left(), current_face_location.top(), current_face_location.right(), current_face_location.bottom()

    print("Found face {} at left: {}, top: {}, right: {}, bottom: {}".format(index + 1, left_x, left_y, right_x, right_y))

    current_face_image = image_to_detect[left_y: right_y, left_x: right_x]

    cv2.imshow("Face No " + str(index + 1), current_face_image)

    cv2.rectangle(image_to_detect, (left_x, left_y), (right_x, right_y), (0, 0, 255), 2)

cv2.imshow("Faces to detect in an image", image_to_detect)

cv2.waitKey(0)

cv2.destroyAllWindows()