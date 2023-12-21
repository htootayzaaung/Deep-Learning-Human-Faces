from deepface import DeepFace
import cv2

# detector_backened = "opencv", "ssd", "dlib", "mtcnn", "retinaface"

face_verified1 = DeepFace.verify("Dataset/Testing/putin11.jpg", "Dataset/Testing/putin12.jpg", detector_backend="opencv")
face_verified2 = DeepFace.verify("Dataset/Testing/putin11.jpg", "Dataset/Testing/trump12.jpg", detector_backend="opencv")

print("putin11.jpg & putin12.jpg: ", face_verified1['verified'])
print("Data-dictionary: ", face_verified1) 

print("putin11.jpg & trump12.jpg: ", face_verified2['verified'])
print("Data-dictionary: ", face_verified2) 