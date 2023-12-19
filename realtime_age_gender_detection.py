import cv2
import face_recognition

# Load the models for gender and age detection
gender_protext = "Models/gender_deploy.prototxt"
gender_caffemodel = "Models/gender_net.caffemodel"
gender_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)

age_protext = "Models/age_deploy.prototxt"
age_caffemodel = "Models/age_net.caffemodel"
age_net = cv2.dnn.readNet(age_caffemodel, age_protext)

gender_list = ['Male', 'Female']
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Initialize webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():
    # Read frame from the webcam
    ret, frame = webcam.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Detect faces using face_recognition library
    face_locations = face_recognition.face_locations(frame, model="hog")

    for face_location in face_locations:
        top, right, bottom, left = face_location

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Extract the face ROI
        face_roi = frame[top:bottom, left:right]

        # Preprocess the face for the gender and age model
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        # Overlay the text on the image
        overlay_text = f"{gender}, {age}"
        cv2.putText(frame, overlay_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Real-time Age and Gender Detection', frame)

    # Press 'q' to break out of the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
webcam.release()
cv2.destroyAllWindows()
