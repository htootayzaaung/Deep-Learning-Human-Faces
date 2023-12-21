from deepface import DeepFace

# detector_backened = "opencv", "ssd", "dlib", "mtcnn", "retinaface"
# model_name = "VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"
# distance_metric = "cosine", "euclidean", "euclidean_l2"

# face recognition
face_recognition = DeepFace.find(img_path="Dataset/Testing/putin11.jpg", db_path="Dataset/Training", model_name="VGG-Face", distance_metric="cosine", detector_backend="opencv")
print(face_recognition)