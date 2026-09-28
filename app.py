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
CLASS_NAMES = {
    0: "Cat",
    1: "Dog"
}

MODEL_PATH = "cat_dog_cnn.keras"


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

    # Resize image
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert image to numpy array
    image_array = np.array(image)

    # Handle grayscale images
    if len(image_array.shape) == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)

    # Handle RGBA images
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array, verbose=0)

    probability = float(prediction[0][0])

    # Sigmoid:
    # probability >= 0.5 → Dog
    # probability < 0.5  → Cat

    if probability >= 0.5:
        predicted_class = "Dog"
        confidence = probability
    else:
        predicted_class = "Cat"
        confidence = 1 - probability

    return predicted_class, confidence


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("🐱 🐶 Cat vs Dog Classifier")

st.write(
    "Upload an image and the CNN model will predict whether it is a **Cat** or **Dog**."
)


# --------------------------------------------------
# File Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Display & Predict
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Selected Image",
        width=400
    )

    if st.button("🔍 Predict"):

        with st.spinner("Analyzing image..."):

            predicted_class, confidence = predict_image(image)

        st.divider()

        st.subheader("Prediction")

        if predicted_class == "Cat":
            st.success(f"🐱 Cat")
        else:
            st.success(f"🐶 Dog")

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.progress(confidence)

        st.write(
            f"The model predicts **{predicted_class}** "
            f"with **{confidence * 100:.2f}% confidence**."
        )

else:

    st.info("👆 Please upload a cat or dog image.")
