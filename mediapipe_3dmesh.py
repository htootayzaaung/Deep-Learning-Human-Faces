import mediapipe as mp
import cv2
from PIL import Image

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Read an image from your directory.
image = cv2.imread('Images/htoo.jpg')

# Convert the color space from BGR (OpenCV) to RGB.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and draw landmarks.
results = face_mesh.process(image_rgb)
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image=image_rgb,  # Draw on the RGB image
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
        )

# Convert the processed image back to BGR color space for consistency with PIL
image_processed = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

# Convert the OpenCV image to a PIL image
pil_image = Image.fromarray(image_processed)

# Display the image using PIL
pil_image.show()

# Save the image using PIL
pil_image.save('Images/htoo_mediapipe.jpg')

# Release resources.
face_mesh.close()
