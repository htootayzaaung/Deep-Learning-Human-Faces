import cv2
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine, euclidean

def detect_extract_face(image_path_to_detect):
    image_to_detect = plt.imread(image_path_to_detect)

    # create an instance of MTCNN detector
    mtcnn_detector = MTCNN()

    # detect all face locations using the mtcnn detector
    all_face_locations = mtcnn_detector.detect_faces(image_to_detect)

    print("There are {} face(s) in this image {}".format(len(all_face_locations), image_path_to_detect))
    # Wait for a key press and then close the window

    # print(all_face_locations)

    image_to_detect = cv2.cvtColor(image_to_detect, cv2.COLOR_BGR2RGB)

    for index, current_face_location in enumerate (all_face_locations):
        x, y, width, height = current_face_location['box']
        left_x, left_y = x, y
        right_x, right_y = x + width, y + height

        #print("Found face {} at top: {}, right: {}, bottom: {}, left: {}".format(index + 1, x, y, x + width, y + height))

        current_face_image = image_to_detect[left_y:right_y, left_x:right_x]

        current_face_image = Image.fromarray(current_face_image)

        current_face_image = current_face_image.resize((224, 224))

        current_face_image_np_array = np.array(current_face_image)

        return current_face_image_np_array

sample_faces = [detect_extract_face("Dataset/Training/Trump/trump1.jpg"),
                detect_extract_face("Dataset/Training/Trump/trump2.jpg"),
                detect_extract_face("Dataset/Training/Trump/trump3.jpg"),
                detect_extract_face("Dataset/Training/Putin/putin1.jpg")]

sample_faces = np.array(sample_faces, 'float32')

sample_faces = preprocess_input(sample_faces, version=2)

vggface_model = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3), pooling='avg')

print("Input shape: {}".format(vggface_model.input_shape))
print(vggface_model.inputs)

sample_face_embeddings = vggface_model.predict(sample_faces)

trump_face1 = sample_face_embeddings[0]
trump_face2 = sample_face_embeddings[1]
trump_face3 = sample_face_embeddings[2]
putin_face1 = sample_face_embeddings[3]

print("\nIf the cosine distance is less than 0.4, then the faces are similar! Elsem they are different!\n")
print("cosine(trump_face1, trump_face2): ", cosine(trump_face1, trump_face2))
print("cosine(trump_face1, trump_face3): ", cosine(trump_face1, trump_face3))
print("cosine(trump_face1, putin_face1): ", cosine(trump_face1, putin_face1))

print("euclidean(trump_face1, trump_face2): ", euclidean(trump_face1, trump_face2))
print("euclidean(trump_face1, trump_face3): ", euclidean(trump_face1, trump_face3))
print("euclidean(trump_face1, putin_face1): ", euclidean(trump_face1, putin_face1))