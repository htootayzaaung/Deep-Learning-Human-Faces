import mediapipe as mp
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Read an image from your directory.
image = cv2.imread('../Images/htoo.jpg')
# Convert the color space from BGR (OpenCV) to RGB.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and get landmarks.
results = face_mesh.process(image_rgb)

# Prepare to plot 3D points.
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Scale factors for x and y axes to maintain the aspect ratio
# This assumes the image width is greater than the image height
scale_x = image.shape[1] / image.shape[0]
scale_y = 1

# If landmarks were detected...
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Initialize a list to store the landmarks.
        x, y = [], []
        for landmark in face_landmarks.landmark:
            # Scale x, y coordinates to maintain the aspect ratio
            x.append(landmark.x * scale_x)
            y.append(landmark.y * scale_y)
        
        # Scatter plot for the landmarks.
        ax.scatter(x, -1 * np.array(y), c='blue', marker='o')  # We're omitting z-axis here
        
        # Set labels and title.
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('3D Facial Landmarks Visualization')

# Rotate the plot to make the face upright
ax.view_init(elev=90, azim=-90)

# Hide the z-axis
ax.set_zticks([])

# Save the plot to a file.
plt.savefig('../Images/htoo_3d_plot_upright.png')

# Show the plot on screen.
plt.show()

# Release resources.
face_mesh.close()
