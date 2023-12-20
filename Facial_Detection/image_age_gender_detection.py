import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition

# loading the image to detect
image_to_detect = cv2.imread("../Images/tyson_francis.jpg")

# Initialize the array variable to hold all the face locations in the frame
all_face_locations = face_recognition.face_locations(image_to_detect, model="hog")

print("There are {} face(s) in this image".format(len(all_face_locations)))


for index, current_face_location in enumerate(all_face_locations):
    top_pos, right_pos, bottom_pos, left_pos = current_face_location

    print("Found face {} at top: {}, right: {}, bottom: {}, left: {}".format(index + 1, top_pos, right_pos, bottom_pos, left_pos))

    # Extract the face from the frame
    current_face_image = image_to_detect[top_pos:bottom_pos, left_pos:right_pos]

    AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    current_face_image_blob = cv2.dnn.blobFromImage(current_face_image, 1, (227, 227), AGE_GENDER_MODEL_MEAN_VALUES, swapRB=False)

    gender_label_list = ['Male', 'Female']
    gender_protext = "Models/gender_deploy.prototxt"
    gender_caffemodel = "Models/gender_net.caffemodel"
    gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
    gender_cov_net.setInput(current_face_image_blob)
    gender_predictions = gender_cov_net.forward()
    gender = gender_label_list[gender_predictions[0].argmax()]

    age_label_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    age_protext = "../Models/age_deploy.prototxt"
    age_caffemodel = "../Models/age_net.caffemodel"
    age_cov_net = cv2.dnn.readNet(age_caffemodel, age_protext)
    age_cov_net.setInput(current_face_image_blob)
    age_predictions = age_cov_net.forward()
    age = age_label_list[age_predictions[0].argmax()]

    cv2.rectangle(image_to_detect, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)

    font = cv2.FONT_HERSHEY_DUPLEX

    cv2.putText(image_to_detect, gender + " " + age + " years", (left_pos, bottom_pos), font, 0.5, (0, 255, 0), 1)


cv2.imshow("Detected Faces with age and gender", image_to_detect)

cv2.waitKey(0)

cv2.destroyAllWindows()
