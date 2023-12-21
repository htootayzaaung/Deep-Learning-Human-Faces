from deepface import DeepFace

# detector_backened = "opencv", "ssd", "dlib", "mtcnn", "retinaface"
# model_name = "VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"
# distance_metric = "cosine", "euclidean", "euclidean_l2"

# face analysis
face_analysis = DeepFace.analyze("Dataset/Testing/putin11.jpg", actions = ['emotion', 'age', 'gender', 'race'])

print("Face Analysis: ", face_analysis)