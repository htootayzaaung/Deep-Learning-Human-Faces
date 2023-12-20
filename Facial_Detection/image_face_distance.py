import cv2
import face_recognition

# Load the image where faces need to be recognized
image_to_recognize_path = "../Images/tyson2.jpg"

# Load and encode faces of Francis Ngannou and Tyson Fury
francis_image = face_recognition.load_image_file("../Images/francis_face.jpg")
francis_face_encodings = face_recognition.face_encodings(francis_image)[0]

tyson_image = face_recognition.load_image_file("../Images/tyson_face.jpg")
tyson_face_encodings = face_recognition.face_encodings(tyson_image)[0]

# Store the encodings and names
known_face_encodings = [francis_face_encodings, tyson_face_encodings]
known_face_names = ["Francis Ngannou", "Tyson Fury"]

image_to_recognize = face_recognition.load_image_file(image_to_recognize_path)
image_to_recognize_encodings = face_recognition.face_encodings(image_to_recognize)[0]

face_distances = face_recognition.face_distance(known_face_encodings, image_to_recognize_encodings)

for i, face_distance in enumerate(face_distances):
    print("The calculated face distance is {:.2} against the sample {}".format(face_distance, known_face_names[i]))
    print("The matching percentage is {}%\n".format(round((1 - face_distance) * 100, 2)))

"""
The convention here is if the distance is less than 0.6, then the faces are considered to be a match.
Else if the distance is greater than 0.6, then the faces are considered to be a mismatch.
The distance of 1 means that the faces are completely different.
"""