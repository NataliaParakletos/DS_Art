import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import requests
import os

def download_model(url, model_path):
    response = requests.get(url)
    with open(model_path, 'wb') as file:
        file.write(response.content)

# Завантаження моделей
cnn_model_url = 'https://drive.google.com/drive/folders/1NNvtnF88qYz-qyq5qmr-NKKe49mVo-Sw'
vgg16_model_url = 'https://drive.google.com/drive/folders/1NNvtnF88qYz-qyq5qmr-NKKe49mVo-Sw'

cnn_model_path = 'cnn_model.h5'
vgg16_model_path = 'vgg16_model.h5'

if not os.path.exists(cnn_model_path):
    download_model(cnn_model_url, cnn_model_path)

if not os.path.exists(vgg16_model_path):
    download_model(vgg16_model_url, vgg16_model_path)

cnn_model = load_model(cnn_model_path)
vgg16_model = load_model(vgg16_model_path)

def predict(model, img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    return predictions

# Інтерфейс Streamlit
st.title("Image Classification Web App")
st.header("Upload an image to classify")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

model_option = st.selectbox("Choose the model", ("CNN", "VGG16"))

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if model_option == "CNN":
        model = cnn_model
    else:
        model = vgg16_model

    st.write("Classifying...")
    predictions = predict(model, img)

    st.write("Predictions:")
    st.write(predictions)

    predicted_class = np.argmax(predictions)
    st.write(f"Predicted class: {predicted_class}")

    # Графіки функції втрат і точності
    try:
        history = model.history.history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history['loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss')
        ax1.legend()

        ax2.plot(history['accuracy'], label='Training Accuracy')
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Accuracy')
        ax2.legend()

        st.pyplot(fig)
    except AttributeError:
        st.write("Training history not available for this model.")
