import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition

# loading the image to detect
image_to_detect = cv2.imread("../Images/tyson_francis.jpg")

# Load the model and load the weights
face_exp_model = model_from_json(open("../Models/facial_expression_model_structure.json", "r").read())

# Load weights into model
face_exp_model.load_weights("../Models/facial_expression_model_weights.h5")

# List of emotion labels
emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Initialize the array variable to hold all the face locations in the frame
all_face_locations = face_recognition.face_locations(image_to_detect, model="hog")

print("There are {} face(s) in this image".format(len(all_face_locations)))


for index, current_face_location in enumerate(all_face_locations):
    top_pos, right_pos, bottom_pos, left_pos = current_face_location

    print("Found face {} at top: {}, right: {}, bottom: {}, left: {}".format(index + 1, top_pos, right_pos, bottom_pos, left_pos))

    # Extract the face from the frame
    current_face_image = image_to_detect[top_pos:bottom_pos, left_pos:right_pos]

    # Convert the extracted face from RGB to Grayscale
    current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
    
    # draw rectangle around face detected
    cv2.rectangle(image_to_detect, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
    
    # Resize the face to 48x48 for the model
    current_face_image = cv2.resize(current_face_image, (48, 48))
    
    # Convert the face to the correct format for the model
    img_pixels = img_to_array(current_face_image)
    
    # Expand the face array
    img_pixels = np.expand_dims(img_pixels, axis=0)
    
    # Normalize the face array to [0, 1]
    img_pixels /= 255
    
    # Predict the emotion of the face
    exp_predictions = face_exp_model.predict(img_pixels)
    
    # Get the emotion with the highest confidence
    max_index = np.argmax(exp_predictions[0])
    emotion_label = emotions_label[max_index]
    
    # Display the predicted emotion on the frame
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_detect, emotion_label, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)

# Show the frame with the detected faces and their predicted emotions
cv2.imshow("Detected Faces and Emotions", image_to_detect)

# Wait indefinitely for a key press before closing the window
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()
