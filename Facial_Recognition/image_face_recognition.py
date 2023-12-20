import cv2
import face_recognition

# Load the image where faces need to be recognized
image_to_recognize = cv2.imread("../Images/tyson_francis.jpg")

# Load and encode faces of Francis Ngannou and Tyson Fury
francis_image = face_recognition.load_image_file("../Images/francis_face.jpg")
francis_face_encodings = face_recognition.face_encodings(francis_image)[0]

tyson_image = face_recognition.load_image_file("../Images/tyson_face.jpg")
tyson_face_encodings = face_recognition.face_encodings(tyson_image)[0]

# Store the encodings and names
known_face_encodings = [francis_face_encodings, tyson_face_encodings]
known_face_names = ["Francis Ngannou", "Tyson Fury"]

# Detect faces in the image
all_face_locations = face_recognition.face_locations(image_to_recognize, model="hog")
all_face_encodings = face_recognition.face_encodings(image_to_recognize, all_face_locations)

print("There are {} face(s) in this image".format(len(all_face_locations)))

# Identify faces in the image
for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    print("Found face at top: {}, right: {}, bottom: {}, left: {}".format(top_pos, right_pos, bottom_pos, left_pos))

    all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
    name_of_person = "Unknown face"

    if True in all_matches:
        first_match_index = all_matches.index(True)
        name_of_person = known_face_names[first_match_index]

    cv2.rectangle(image_to_recognize, (left_pos, top_pos), (right_pos, bottom_pos), (255, 0, 0), 2)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_recognize, name_of_person, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)

# Display the result
cv2.imshow("Identified Faces", image_to_recognize)
cv2.waitKey(0)
cv2.destroyAllWindows()
