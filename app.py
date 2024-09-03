import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
@st.cache_resource
def load_model():
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter
def preprocess_image(image_path):
    img = Image.open(image_path).resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img
def predict(image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output
interpreter = load_model()
# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
st.title("A/I vs Real Image classifier")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = preprocess_image(uploaded_file)
    prediction = predict(image)
    
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write(f"Prediction:")
    st.write(f"A.I : {prediction[0][0] * 100 : .2f}%")
    st.write(f"Real : {prediction[0][1] * 100 : .2f}%")