import streamlit as st
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model

# Load labels from a file
def load_labels(label_file_path):
    with open(label_file_path, 'r') as f:
        labels = f.read().splitlines()
    return labels

# Function to predict bird name from image
def predict_bird(image_path, model, labels):
    # Load the image using cv2
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load the image using Keras
    img_keras = load_img(image_path, target_size=(224, 224))
    img_keras = img_to_array(img_keras)
    img_keras = np.expand_dims(img_keras, axis=0)
    
    # Predict using your model
    pred = model.predict(img_keras)
    
    # Get the index of the predicted class
    s = np.argmax(pred)
    
    # Get the predicted bird name
    bird_name = labels[s]
    
    return img, bird_name

# Load your model and labels here
model_path = r'C:/Users/gopi naga priya/project/nnsavedmodels/chillimodelpred72.h5'  # Replace with the path to your trained model
label_file_path = 'bird_names.txt'  # Replace with the path to your labels file
model = load_model(model_path)
labels = load_labels(label_file_path)

# Streamlit UI
st.title("Bird Species Prediction")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Save the uploaded image to a temporary file
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Create a button to trigger prediction
    if st.button("Predict"):
        # Predict the bird name
        img, bird_name = predict_bird("temp.jpg", model, labels)
        
        # Display the predicted bird name
        st.write(f"Predicted Bird: {bird_name}")
        
        # Display the image
        st.image(img, caption="Uploaded Image", use_column_width=True)

# Run the app with: streamlit run your_script.py
