import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from keras.models import load_model

model = load_model('hair_length_classifier.keras')
model1 = load_model('Detection.keras')

def Detect(image):
    image = image.resize((48, 48))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = np.delete(image, 0, 1)
    image = np.resize(image, (48, 48, 3))
    image = np.array([image]) / 255
    pred = model1.predict(image)
    age = int(np.round(pred[1][0]))
    sex_f = ["Male", "Female"]
    sex = int(np.round(pred[0][0]))
    return age, sex_f[sex]

img_height, img_width = 128, 128
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img = image.resize((img_height, img_width))
    img = img_to_array(img) / 255.0 
    img = np.expand_dims(img, axis=0) 
    return img

def predict_hair_length(image):
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)
    if prediction[0][0] > 0.5:
        return "Long Hair", prediction[0][0]
    else:
        return "Short Hair", prediction[0][0]
    
st.title("Long Hair Identification")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Processing...")

    if st.button("Predict"):
        age, gender = Detect(image)
        label, confidence = predict_hair_length(image)
        
        if 20 <= age <= 30:
            if gender == "Male" and label == "Long Hair":
                gender = "Female"
            elif gender == "Female" and label == "Short Hair":
                gender = "Male"

        st.write(f"Predicted Gender: {gender}")

