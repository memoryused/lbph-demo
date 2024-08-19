import cv2
import pickle
import numpy as np

# Load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("lbph_face_recognizer.xml")

# Load the label map
with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

# Function to predict the label of an image
def predict_image(image_path, img_size=(200, 200)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to read")
    
    # Resize the image to match the training size
    image = cv2.resize(image, img_size)

    label, confidence = recognizer.predict(image)
    return label, confidence

# Function to calculate accuracy on the test set
def calculate_accuracy(images, labels):
    correct_predictions = 0
    for i, image in enumerate(images):
        label, confidence = recognizer.predict(image)
        if label == labels[i]:
            correct_predictions += 1
    accuracy = correct_predictions / len(images)
    return accuracy

# Load test set
test_images = np.load("test_images.npy")
test_labels = np.load("test_labels.npy")

# Calculate and print test accuracy
test_accuracy = calculate_accuracy(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
