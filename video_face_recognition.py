import cv2
import face_recognition

# Capture the video from a file
webcam_video_stream = cv2.VideoCapture('Videos/tyson_fury.mp4')

# Load and encode faces of Francis Ngannou and Tyson Fury
francis_image = face_recognition.load_image_file("Images/francis_face.jpg")
francis_face_encodings = face_recognition.face_encodings(francis_image)[0]

tyson_image = face_recognition.load_image_file("Images/tyson_face.jpg")
tyson_face_encodings = face_recognition.face_encodings(tyson_image)[0]

htoo_image = face_recognition.load_image_file("Images/htoo.jpg")
htoo_face_encodings = face_recognition.face_encodings(htoo_image)[0]

# Store the encodings and names
known_face_encodings = [francis_face_encodings, tyson_face_encodings, htoo_face_encodings]
known_face_names = ["Francis Ngannou", "Tyson Fury", "Htoo"]

# Initialize variables to hold all the face locations, encodings, and names in the frame
all_face_locations = []
all_face_encodings = []
all_face_names = []

# Loop through every frame in the video
while True:
    # Get the current frame from the video stream as an image
    ret, current_frame = webcam_video_stream.read()

    # Check if the frame is returned, if not, reset to the beginning of the video
    if not ret:
        webcam_video_stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Resize the current frame to 1/4 size to process faster
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)

    # Detect all faces in the image
    all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=1, model='hog')
    
    # Detect face encodings for all the faces detected
    all_face_encodings = face_recognition.face_encodings(current_frame_small, all_face_locations)

    # Loop through the face locations and the face embeddings
    for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
        # Splitting the tuple to get the four position values of current face
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        
        # Change the position magnitude to fit the actual size video frame
        top_pos *= 4
        right_pos *= 4
        bottom_pos *= 4
        left_pos *= 4
        
        # Find all the matches and get the list of matches
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
       
        # String to hold the label
        name_of_person = 'Unknown face'
        
        # Check if the all_matches have at least one item
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]
        
        # Draw rectangle around the face    
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (255, 0, 0), 2)
        
        # Display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)
    
    # Display the video
    cv2.imshow("Webcam Video", current_frame)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the stream and camera, close all OpenCV windows
webcam_video_stream.release()
cv2.destroyAllWindows()
