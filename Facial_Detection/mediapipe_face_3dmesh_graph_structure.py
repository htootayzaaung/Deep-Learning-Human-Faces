import mediapipe as mp
import cv2
from PIL import Image, ImageDraw

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Read an image from your directory.
image_path = '../Images/htoo.jpg'  # Make sure this path is correct.
image = cv2.imread(image_path)
h, w, _ = image.shape

# Convert the color space from BGR (OpenCV) to RGB.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and get landmarks.
results = face_mesh.process(image_rgb)

# Overlay landmarks on the RGB image.
if results.multi_face_landmarks:
    # Convert the MediaPipe landmark data to a PIL image for drawing
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # Set the width for the mesh lines
    line_width = 2  # Increase the width as needed

    # Iterate over the detected faces
    for face_landmarks in results.multi_face_landmarks:
        # Draw the points
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            draw.point((x, y), 'green')

        # Draw the mesh lines
        for connection in mp_face_mesh.FACEMESH_TESSELATION:
            start_idx = connection[0]
            end_idx = connection[1]
            start_point = face_landmarks.landmark[start_idx]
            end_point = face_landmarks.landmark[end_idx]
            start_point = (int(start_point.x * w), int(start_point.y * h))
            end_point = (int(end_point.x * w), int(end_point.y * h))
            draw.line([start_point, end_point], fill='green', width=line_width)

# Save the image with landmarks and mesh using PIL.
output_image_path = '../Images/htoo_with_face_mesh.jpg'
pil_image.save(output_image_path)

# Display the image using PIL.
pil_image.show()
