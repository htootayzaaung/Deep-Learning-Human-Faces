import cv2
import face_recognition

image = cv2.imread("../Images/tyson_francis.jpg")
cv2.imshow("Tyson Fury Vs Francis Nagannou", image)

all_face_locations = face_recognition.face_locations(image, model="hog")

# CNN
# all_face_locations = face_recognition.face_locations(image, model="cnn")


print("There are {} face(s) in this image".format(len(all_face_locations)))
# Wait for a key press and then close the window

for index, current_face_location in enumerate (all_face_locations):
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    print("Found face {} at top: {}, right: {}, bottom: {}, left: {}".format(index + 1, top_pos, right_pos, bottom_pos, left_pos))
    current_face_image = image[top_pos:bottom_pos, left_pos:right_pos]
    cv2.imshow("Face no" + str(index + 1), current_face_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
