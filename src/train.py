print("Starting training...")  # Debugging line

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from data_preprocessing import preprocess_and_split_data


# Define Model
def create_model():
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(6, activation="softmax"),  # Adjust based on number of classes
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# Train Model
def train_model():
    print("Creating model...")  # Debugging line
    model = create_model()

    # Use real dataset from data_preprocessing.py
    train_data, X_val, y_val = preprocess_and_split_data()

    print("Starting model training...")  # Debugging line
    model.fit(train_data, epochs=10, validation_data=(X_val, y_val))

    print("Saving model...")  # Debugging line
    model.save("models/cell_classifier.keras")


if __name__ == "__main__":
    train_model()
    print("Training completed successfully!")  # Debugging line
