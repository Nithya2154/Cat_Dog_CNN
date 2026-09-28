import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱",
    layout="centered"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMG_SIZE = 128

MODEL_PATH = "cat_dog_cnn.keras"

CONFIDENCE_THRESHOLD = 0.70


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert image to NumPy array
    image_array = np.array(image)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )

    probability = float(prediction[0][0])

    # Determine class
    if probability >= 0.5:

        predicted_class = "Dog"
        confidence = probability

    else:

        predicted_class = "Cat"
        confidence = 1 - probability


    # --------------------------------------------------
    # Confidence Check
    # --------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        predicted_class = "No Cat or Dog"
        confidence = confidence


    return predicted_class, confidence


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("🐱 🐶 Cat vs Dog Classifier")

st.write(
    "Upload an image and the CNN model will predict "
    "whether it is a **Cat** or **Dog**."
)


# --------------------------------------------------
# File Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Display Image
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Selected Image",
        width=400
    )


    # --------------------------------------------------
    # Prediction Button
    # --------------------------------------------------

    if st.button("🔍 Predict"):

        with st.spinner("Analyzing image..."):

            predicted_class, confidence = predict_image(image)


        st.divider()

        st.subheader("Prediction")


        # --------------------------------------------------
        # Display Result
        # --------------------------------------------------

        if predicted_class == "Cat":

            st.success("🐱 Cat detected")

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )


        elif predicted_class == "Dog":

            st.success("🐶 Dog detected")

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )


        else:

            st.warning(
                "❌ No Cat or Dog detected with sufficient confidence."
            )

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )


        # --------------------------------------------------
        # Confidence Bar
        # --------------------------------------------------

        st.progress(confidence)


else:

    st.info("👆 Please upload a cat or dog image.")
```
