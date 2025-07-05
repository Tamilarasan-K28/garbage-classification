import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model("model/waste_classifier_model.h5")
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

st.title("Garbage Classification AI App")
uploaded_file = st.file_uploader("Upload a Waste Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    label = class_names[np.argmax(prediction)]
    st.success(f"Predicted Class: {label}")
