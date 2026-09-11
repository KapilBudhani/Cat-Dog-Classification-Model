import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = 128

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cat_dog_model.keras")

model = load_model()

st.title("CNN - Cat vs Dog Classifier")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "gif"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=400)

    if st.button("Predict"):
        processed_image = image.resize((IMG_SIZE, IMG_SIZE))
        processed_image = np.asarray(
            processed_image,
            dtype=np.float32
        ) / 255.0

        processed_image = np.expand_dims(
            processed_image,
            axis=0
        )

        probability = model.predict(
            processed_image,
            verbose=0
        )[0][0]

        if probability >= 0.5:
            label = "Dog"
            confidence = probability
        else:
            label = "Cat"
            confidence = 1 - probability

        st.subheader("Prediction")
        st.write(f"**Class:** {label}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")