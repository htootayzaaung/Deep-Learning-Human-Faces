import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import model_from_json
import face_recognition

# Capture the video from the default camera
webcam_video_stream = cv2.VideoCapture(0)

# Load the model and load the weights
face_exp_model = model_from_json(open("../Models/facial_expression_model_structure.json", "r").read())

# Load weights into model
face_exp_model.load_weights("../Models/facial_expression_model_weights.h5")

# List of emotion labels
emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Initialize the array variable to hold all the face locations in the frame
all_face_locations = []

# Loop through every frame in the video
while True:
    # Get the current frame from the video stream as an image
    ret, current_frame = webcam_video_stream.read()
    
    # Resize the current frame to 1/4 size to process faster
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)
    
    # Detect all faces in the image. Increasing the parameter for number_of_times_to_unsample will improve the ability to detect faces from a distance.
    all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=2, model="hog")

    for index, current_face_location in enumerate(all_face_locations):
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        top_pos = top_pos * 4
        right_pos = right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4

        print("Found face {} at top: {}, right: {}, bottom: {}, left: {}".format(index + 1, top_pos, right_pos, bottom_pos, left_pos))

        # Extract the face from the frame
        current_face_image = current_frame[top_pos:bottom_pos, left_pos:right_pos]
        
        # Convert the face to grayscale
        current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
        
        # Resize the face to 48x48 for the model
        current_face_image = cv2.resize(current_face_image, (48, 48))
        
        # Convert the face to the correct format for the model
        img_pixels = img_to_array(current_face_image)
        
        # Expand the face array
        img_pixels = np.expand_dims(img_pixels, axis=0)
        
        # Normalize the face array to [0, 1]
        img_pixels /= 255

        # Draw a rectangle around the face
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
        
        # Predict the emotion of the face
        exp_predictions = face_exp_model.predict(img_pixels)
        
        # Get the emotion with the highest confidence
        max_index = np.argmax(exp_predictions[0])
        emotion_label = emotions_label[max_index]
        
        # Display the predicted emotion on the frame
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, emotion_label, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)

    # Show the frame with the detected faces and their predicted emotions
    cv2.imshow("Web-Cam Video", current_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
webcam_video_stream.release()
cv2.destroyAllWindows()
