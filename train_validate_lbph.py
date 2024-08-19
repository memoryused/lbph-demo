import cv2
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# Function to read images and labels from the dataset directory
def read_images_and_labels(dataset_path, img_size=(200, 200)):
    images = []
    labels = []
    label_map = {}
    label_counter = 0

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            # Resize the image to ensure all images have the same dimensions
            image = cv2.resize(image, img_size)

            if person_name not in label_map:
                label_map[person_name] = label_counter
                label_counter += 1

            label = label_map[person_name]
            images.append(image)
            labels.append(label)

    return images, labels, label_map

# Function to augment data
def augment_data(images, labels):
    augmented_images = []
    augmented_labels = []

    for image, label in zip(images, labels):
        augmented_images.append(image)
        augmented_labels.append(label)

        # Flip horizontally
        flipped_image = cv2.flip(image, 1)
        augmented_images.append(flipped_image)
        augmented_labels.append(label)

        # Rotate
        rows, cols = image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)
        rotated_image = cv2.warpAffine(image, M, (cols, rows))
        augmented_images.append(rotated_image)
        augmented_labels.append(label)

        # Scale
        scaled_image = cv2.resize(image, (int(cols * 1.2), int(rows * 1.2)))
        scaled_image = cv2.resize(scaled_image, (cols, rows))
        augmented_images.append(scaled_image)
        augmented_labels.append(label)

    return augmented_images, augmented_labels

# Read images and labels
dataset_path = "dataset"
images, labels, label_map = read_images_and_labels(dataset_path)

# Augment data
images, labels = augment_data(images, labels)

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split the dataset into train and validation sets (70:30)
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.3, random_state=42)

# Create the LBPH face recognizer with tuned parameters
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

# Function to calculate accuracy and custom entropy loss
def calculate_metrics(images, labels, num_classes):
    correct_predictions = 0
    log_loss_sum = 0
    epsilon = 1e-15  # Small value to prevent log(0)

    for i, image in enumerate(images):
        label, confidence = recognizer.predict(image)
        correct_predictions += (label == labels[i])

        # Simulate probability distributions for custom log loss
        predicted_prob = np.ones(num_classes) * (1 - confidence) / (num_classes - 1)
        predicted_prob[label] = confidence
        predicted_prob = np.clip(predicted_prob, epsilon, 1 - epsilon)
        predicted_prob /= predicted_prob.sum()

        true_prob = np.zeros(num_classes)
        true_prob[labels[i]] = 1

        log_loss_sum += -np.sum(true_prob * np.log(predicted_prob))

    accuracy = correct_predictions / len(images)
    entropy_loss = log_loss_sum / len(images)
    return accuracy, entropy_loss

# Custom training loop with logging
num_epochs = 10  # Define the number of epochs
num_classes = len(np.unique(labels))  # Get the number of unique classes
for epoch in range(num_epochs):
    recognizer.train(train_images, train_labels)
    
    train_accuracy, train_loss = calculate_metrics(train_images, train_labels, num_classes)
    val_accuracy, val_loss = calculate_metrics(val_images, val_labels, num_classes)
    
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Training Loss (Entropy): {train_loss:.4f}")
    print(f"Validation Loss (Entropy): {val_loss:.4f}")
    print("----------------------------")

# Save the trained model and label map
recognizer.save("lbph_face_recognizer.xml")
with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)

# Save validation set for later evaluation
np.save("val_images.npy", val_images)
np.save("val_labels.npy", val_labels)
