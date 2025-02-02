import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("models/cell_classifier.h5")

# Inference Function
def predict_cell(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction.argmax()
