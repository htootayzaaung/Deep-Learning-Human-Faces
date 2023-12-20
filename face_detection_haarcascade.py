import cv2

image_to_detect = cv2.imread("Images/tyson_francis.jpg")

haar_cascade_path = "Models/haarcascades/haarcascade_frontalface_default.xml"
face_detection_classifier = cv2.CascadeClassifier(haar_cascade_path)

all_face_locations = face_detection_classifier.detectMultiScale(image_to_detect)

print("There are {} face(s) in this image".format(len(all_face_locations)))
# Wait for a key press and then close the window

for index, current_face_location in enumerate (all_face_locations):
    x, y, width, height = current_face_location
    left_x, left_y = x, y
    right_x, right_y = x + width, y + height

    print("Found face {} at top: {}, right: {}, bottom: {}, left: {}".format(index + 1, x, y, x + width, y + height))

    current_face_image = image_to_detect[left_y:right_y, left_x:right_x]   

    cv2.imshow("Face no" + str(index + 1), current_face_image)

    cv2.rectangle(image_to_detect, (x, y), (x + width, y + height), (0, 0, 255), 2)

cv2.imshow("Image with faces", image_to_detect)
cv2.waitKey(0)
