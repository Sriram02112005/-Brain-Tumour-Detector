import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Ensemble_functions import output_label, ensemble_output, image_preprocess, load_models
from PIL import Image
import os

st.title('🧠 NeuroVision')
st.write('An approach that incorporates the power of deep learning Ensembles to identify Brain Tumours from the images')
st.subheader("Description")
st.write("")
col1,col2,col3,col4 = st.columns([1,5,5,1])

# Glioma
with col2:
    with st.expander("1. Glioma"):
        st.markdown('''
    <h4>Description:</h4> 
    Gliomas are a type of tumor that occurs in the brain and spinal cord. These tumors originate from glial cells, which support and protect neurons. Gliomas can vary in aggressiveness, ranging from low-grade (slow-growing) to high-grade (fast-growing and malignant).
    
    <h4>Common Symptoms:</h4> 
    Headaches, seizures, nausea, memory problems, or speech issues, depending on the location of the tumor.
    
    <h4>Treatment:</h4> 
    Treatment options include surgery, radiation therapy, and chemotherapy.
    ''', unsafe_allow_html=True)

# Meningioma
with col3:
    with st.expander("2. Meningioma"):
        st.markdown('''
    <h4>Description:</h4> 
    Meningiomas are tumors that develop in the meninges, the protective layers of tissue covering the brain and spinal cord. Most meningiomas are benign (non-cancerous), though they can cause significant problems if they grow large enough to press on the brain.
    
    <h4>Common Symptoms:</h4> 
    Headaches, vision problems, hearing loss, or seizures, depending on the tumor's location.
    
    <h4>Treatment:</h4> 
    Treatment often involves surgery, and in some cases, radiation therapy may be used.
    ''', unsafe_allow_html=True)

# Pituitary Tumor
with col2:
    with st.expander("3. Pituitary Tumor"):
        st.markdown('''
    <h4>Description:</h4> 
    Pituitary tumors form in the pituitary gland, located at the base of the brain. These tumors can affect hormone production, leading to various symptoms. Most pituitary tumors are benign and can be functioning (hormone-producing) or non-functioning.
    
    <h4>Common Symptoms:</h4> 
    Hormonal imbalances, vision problems, fatigue, or unexplained weight changes.
    
    <h4>Treatment:</h4> 
    Treatment includes medication to manage hormone levels, surgery to remove the tumor, and sometimes radiation therapy.
    ''', unsafe_allow_html=True)

# No Tumor
with col3:
    with st.expander("4. No Tumor"):
        st.markdown('''
    <h4>Description:</h4> 
    This category indicates that the brain scan does not show the presence of any tumor. The image appears to be free from abnormalities related to brain tumors.
    
    <h4>Implications:</h4> 
    While this category suggests no tumor, it's important for users to seek medical advice to confirm the results and ensure no other neurological conditions are present.
    ''', unsafe_allow_html=True)

st.divider()
st.header('Upload the image')
file = st.file_uploader(label='Image file',
                 label_visibility='hidden',
                 type=['png', 'jpg','jpeg'])

model_input = 0
if file is not None:
    image = Image.open(file)
    image_array = np.array(image)
    preprocessed_img = image_preprocess(image_array)
    preprocessed_img_np = preprocessed_img.numpy()
    model_input = preprocessed_img_np
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        st.image(preprocessed_img_np, width=350)

st.write("")
left, middle, right = st.columns([1,1,1])
with middle:
    click = st.button('Predict', use_container_width=True)

output_arr = 0
class_name_predicted = ""

# Load models once at startup (more efficient)
@st.cache_resource
def load_all_models():
    """Load all models and cache them"""
    try:
        densenet = tf.keras.models.load_model("densenet169_model.keras")
        vgg19 = tf.keras.models.load_model("VGG19_model.keras")
        xception = tf.keras.models.load_model("xception_model.keras")
        effnet = tf.keras.models.load_model("EfficientNetV2B2_model.keras")
        return densenet, vgg19, xception, effnet
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

if click:
    if file is None:
        st.error("Please upload an image first!")
    else:
        with st.spinner('Loading models...'):
            densenet, vgg19, xception, effnet = load_all_models()
            
        if densenet is not None:
            st.success("Models loaded successfully")
            
            st.subheader('Output Probability')
            with st.spinner('Predicting.....'):
                output_arr, class_name_predicted = ensemble_output(model_input, densenet, vgg19, xception, effnet)
            
            with st.expander("View Output Probabilities"):
                st.metric(label="Glioma", value=f"{output_arr[0] * 100:.2f}%")
                st.progress(float(output_arr[0]))
                st.metric(label="Meningioma", value=f"{output_arr[1] * 100:.2f}%")
                st.progress(float(output_arr[1]))
                st.metric(label="No Tumour", value=f"{output_arr[2] * 100:.2f}%")
                st.progress(float(output_arr[2]))
                st.metric(label="Pituitary", value=f"{output_arr[3] * 100:.2f}%")
                st.progress(float(output_arr[3]))
                st.subheader(" ")
            
            st.markdown(f'''
            <div style="
                background-color: #d4edda;
                color: #155724;
                padding: 10px;
                border-radius: 5px;
                font-size: 24px;
                border: 1px solid #c3e6cb;
                ">
                <strong>Prediction:</strong>  {class_name_predicted}
            </div>
            ''', unsafe_allow_html=True)
