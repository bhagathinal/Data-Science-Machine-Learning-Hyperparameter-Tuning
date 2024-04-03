import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the pre-trained MNIST model
model = load_model('mnist_model.h5')

# Function to preprocess the input image
def preprocess_image(image):
    # Resize image to 28x28 pixels
    image = image.resize((28, 28))
    # Convert image to grayscale
    image = image.convert('L')
    # Convert image to numpy array
    image_array = np.array(image)
    # Normalize pixel values
    image_array = image_array / 255.0
    # Reshape array to match model input shape
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.expand_dims(image_array, axis=-1)
    return image_array

# Set title
st.title('Handwritten Digit prediction')

# Example of adding a file uploader to upload images
uploaded_file = st.file_uploader('Upload Image', type=['jpg', 'png'])

if uploaded_file is not None:
    # Read the uploaded image
    img = Image.open(uploaded_file)

    # Preprocess the input image
    processed_img = preprocess_image(img)

    # Make prediction
    prediction = model.predict(processed_img)
    predicted_digit = np.argmax(prediction)

    # Show uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Show prediction result
    st.markdown(f'<p style="font-size:24px; font-weight:bold;">Predicted Digit: {predicted_digit}</p>', unsafe_allow_html=True)
