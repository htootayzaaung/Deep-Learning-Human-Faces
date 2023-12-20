import face_recognition
import cv2

# Capture the video from the default camera
webcam_video_stream = cv2.VideoCapture(0)

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

# Initialize the array varaible to hold all the face locations in the frame
all_face_locations = []
all_face_encodings = []
all_face_names = []

# loop through every frame in the video
while True:
    # get the current frame from the video stream as an image
    ret, current_frame = webcam_video_stream.read()

    # resize the current frame to 1/4 size to process faster
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)

    # detect all faces in the image

    # Increasing the parameter for number_of_times_to_unsample will improve the ability to detect faces in a further distance
    # By default, its value is 1 and will freeze if the faces is above certain distance in threshold
    all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample = 1, model="hog")

    all_face_encodings = face_recognition.face_encodings(current_frame_small, all_face_locations)
    
    #looping through the face locations and the face embeddings
    for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
        #splitting the tuple to get the four position values of current face
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        
        #change the position maginitude to fit the actual size video frame
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        
        #find all the matches and get the list of matches
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
       
        #string to hold the label
        name_of_person = 'Unknown face'
        
        #check if the all_matches have at least one item
        #if yes, get the index number of face that is located in the first index of all_matches
        #get the name corresponding to the index number and save it in name_of_person
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]
        
        #draw rectangle around the face    
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
        
        #display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)

    cv2.imshow("Web-Cam Video", current_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
webcam_video_stream.release()
cv2.destroyAllWindows()

