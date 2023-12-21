from deepface import DeepFace
import cv2

# detector_backened = "opencv", "ssd", "dlib", "mtcnn", "retinaface"

# face detection and alignment
face_detected = DeepFace.extract_faces(img_path="Dataset/Testing/putin11.jpg", detector_backend="opencv")[0]["face"]

#cv2.cvtColor(face_detected, cv2.COLOR_BGR2RGB)
cv2.imshow("Face Detected", face_detected)

cv2.waitKey(0)
cv2.destroyAllWindows()