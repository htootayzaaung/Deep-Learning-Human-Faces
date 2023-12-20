import face_recognition
from PIL import Image, ImageDraw

# Load the image file
face_image_path = "../Images/tyson_francis.jpg"
face_image = face_recognition.load_image_file(face_image_path)
face_landmarks_list = face_recognition.face_landmarks(face_image)

# Create a PIL image object for drawing
pil_image = Image.fromarray(face_image)
draw = ImageDraw.Draw(pil_image)

# Check if any face landmarks were found and draw them
if face_landmarks_list:
    for i, face_landmarks in enumerate(face_landmarks_list):
        # Print the landmarks for the current face
        print(f"All the points for landmarks in face {i+1}:\n", face_landmarks)

        # Draw lines for all features for the current face
        for facial_feature in face_landmarks.keys():
            draw.line(face_landmarks[facial_feature], width=5, fill='green')

        # Draw dots for all points for each feature for the current face
        dot_size = 2
        for facial_feature in face_landmarks.keys():
            for point in face_landmarks[facial_feature]:
                draw.ellipse((point[0] - dot_size, point[1] - dot_size,
                              point[0] + dot_size, point[1] + dot_size), fill='red')

# Save the image with all landmarks for all faces
all_landmarks_image_path = "../Images/tyson_francis_landmarks.jpg"
pil_image.save(all_landmarks_image_path)
pil_image.show()

print("Image with all landmarks saved to:", all_landmarks_image_path)
