import numpy as np
import streamlit as st
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Load the trained model
mnist_model = load_model('mnist_model.h5')

# Predict class probabilities for the test set
predicted_probs = mnist_model.predict(X_test.reshape(-1, 28, 28, 1))

# Get the predicted classes
predicted_classes = np.argmax(predicted_probs, axis=1)

# Find correct and incorrect predictions
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

# Display heading and counts
st.title("MNIST Digit Classifier Evaluation")
st.write(len(correct_indices), "classified correctly")
st.write(len(incorrect_indices), "classified incorrectly")
st.header("Correct Predictions:")

# Plot correct predictions
correct_columns = st.columns(3)
for i, correct in enumerate(correct_indices[:9]):
    with correct_columns[i % 3]:
        plt.figure(figsize=(5, 5))
        plt.imshow(X_test[correct], cmap='gray', interpolation='none')
        plt.title("Predicted: {}\nTruth: {}".format(predicted_classes[correct], y_test[correct]), fontsize=20, loc='center')
        plt.xticks([])
        plt.yticks([])
        st.pyplot(plt)

# Display heading for incorrect predictions
st.header("Incorrect Predictions:")

# Plot incorrect predictions
incorrect_columns = st.columns(3)
for i, incorrect in enumerate(incorrect_indices[:9]):
    with incorrect_columns[i % 3]:
        plt.figure(figsize=(5, 5))
        plt.imshow(X_test[incorrect], cmap='gray', interpolation='none')
        plt.title("Predicted: {}\nTruth: {}".format(predicted_classes[incorrect], y_test[incorrect]), fontsize=20, loc='center')
        plt.xticks([])
        plt.yticks([])
        st.pyplot(plt)

