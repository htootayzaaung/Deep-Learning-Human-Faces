import face_recognition
from PIL import Image, ImageDraw

# Load the image file
face_image_path = "../Images/htoo.jpg"
face_image = face_recognition.load_image_file(face_image_path)
face_landmarks_list = face_recognition.face_landmarks(face_image)

# Print out all the points for landmarks in the face
print("All the points for landmarks in the face:\n", face_landmarks_list)

# Check if any face landmarks were found
if face_landmarks_list:
    # Draw the main facial features (9 dots)
    pil_image_main_features = Image.fromarray(face_image)
    draw_main = ImageDraw.Draw(pil_image_main_features)

    # Specify the main features to draw
    main_features = [
        'chin',
        'left_eyebrow',
        'right_eyebrow',
        'nose_bridge',
        'nose_tip',
        'left_eye',
        'right_eye',
        'top_lip',
        'bottom_lip'
    ]

    # Draw all facial landmarks (detailed features)
    pil_image_all_landmarks = Image.fromarray(face_image)
    draw_all = ImageDraw.Draw(pil_image_all_landmarks)

    # Draw lines for all features
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            draw_all.line(face_landmarks[facial_feature], width=5, fill='green')

    # Draw dots for all points
    dot_size = 2
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            for point in face_landmarks[facial_feature]:
                draw_all.ellipse((point[0] - dot_size, point[1] - dot_size, 
                                  point[0] + dot_size, point[1] + dot_size), fill='red')

    # Save the image with all landmarks
    all_landmarks_image_path = "../Images/htoo_all_landmarks.jpg"
    pil_image_all_landmarks.save(all_landmarks_image_path)
    pil_image_all_landmarks.show()

print("Image with all landmarks saved to:", all_landmarks_image_path)
