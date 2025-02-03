import os
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load dataset
# Define dataset paths
LABELS_FILE = "./data/labels.csv"
IMAGE_DIR = "./data/images/"


def load_data():
    # Load labels
    df = pd.read_csv(LABELS_FILE)

    images, labels = [], []

    for index, row in df.iterrows():
        image_name = f"BloodImage_{int(row['Image']):05d}.jpg"  # Convert number to 5-digit format
        image_path = os.path.join(IMAGE_DIR, image_name)

        # label = row["Category"]  # Ensure column exists in CSV
        # Create a mapping of categories to numbers
        categories = df["Category"].unique()
        category_to_num = {cat: i for i, cat in enumerate(categories)}
        label = category_to_num[row["Category"]]

        # Load and preprocess the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            continue

        img = cv2.resize(img, (128, 128))  # Resize to match CNN input
        img = img / 255.0  # Normalize pixel values to [0,1]

        images.append(img)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


# Data Augmentation
def augment_data(X_train, y_train):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )
    return datagen.flow(X_train, y_train, batch_size=32)


def print_categories():
    df = pd.read_csv(LABELS_FILE)
    unique_categories = df["Category"].unique()
    print("Unique categories found:", unique_categories)
    print("Number of unique categories:", len(unique_categories))

    # Create a mapping of categories to numbers starting from 0
    category_to_num = {cat: i for i, cat in enumerate(sorted(unique_categories))}
    print("Category mapping:", category_to_num)


# Main function to preprocess data
def preprocess_and_split_data():

    print_categories()

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    augmented_data = augment_data(X_train, y_train)

    print(f"Dataset loaded: {len(X_train)} train images, {len(X_test)} test images")

    return augmented_data, X_test, y_test


# Run script manually (if needed)
if __name__ == "__main__":
    preprocess_and_split_data()
