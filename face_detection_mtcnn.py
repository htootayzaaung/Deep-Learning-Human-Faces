import cv2
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt

image_to_detect = plt.imread("Images/tyson_francis.jpg")

# create an instance of MTCNN detector
mtcnn_detector = MTCNN()

# detect all face locations using the mtcnn detector
all_face_locations = mtcnn_detector.detect_faces(image_to_detect)

print("There are {} face(s) in this image".format(len(all_face_locations)))
# Wait for a key press and then close the window

image_to_detect = cv2.cvtColor(image_to_detect, cv2.COLOR_BGR2RGB)

for index, current_face_location in enumerate (all_face_locations):
    x, y, width, height = current_face_location['box']
    left_x, left_y = x, y
    right_x, right_y = x + width, y + height

    print("Found face {} at top: {}, right: {}, bottom: {}, left: {}".format(index + 1, x, y, x + width, y + height))

    current_face_image = image_to_detect[left_y:right_y, left_x:right_x]   

    cv2.imshow("Face no" + str(index + 1), current_face_image)

    cv2.rectangle(image_to_detect, (x, y), (x + width, y + height), (0, 0, 255), 2)

    keypoints = current_face_location['keypoints']
    cv2.circle(image_to_detect, (keypoints['left_eye']), 5, (0, 255, 0), 1)
    cv2.circle(image_to_detect, (keypoints['right_eye']), 5, (0, 255, 0), 1)
    cv2.circle(image_to_detect, (keypoints['nose']), 5, (0, 255, 0), 1)
    cv2.circle(image_to_detect, (keypoints['mouth_left']), 5, (0, 255, 0), 1)
    cv2.circle(image_to_detect, (keypoints['mouth_right']), 5, (0, 255, 0), 1)

cv2.imshow("Image with faces", image_to_detect)
cv2.waitKey(0)
