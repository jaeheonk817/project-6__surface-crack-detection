#!/usr/bin/env python
# coding: utf-8

# # Deployment using Streamlit
# This file's sole purpose is to use streamlit to make a deployment application.

# In[2]:


# Make Imports
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os


# In[3]:


# Load the pre-trained model
model = tf.keras.models.load_model('models/fifth_model.h5')

# Define a function to make predictions on uploaded images
def predict(image):
    img = Image.open(image)
    img = img.resize((128,128))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction[0][0]

# Define the Streamlit app
def app():
    st.title("Image Classification App by Jae\nIdentifying Cracks in Wall Images")
    st.write("Upload an image and get a prediction of whether the wall has a crack or not")
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file:
        image = Image.open(file)
        st.image(image, caption='Uploaded a wall image', use_column_width=True)
        prediction = predict(file)
        if prediction > 0.5:
            st.write("Prediction: CRACK")
        else:
            st.write("Prediction: NO CRACK")

if __name__ == '__main__':
    app()


# jupyter nbconvert --to python /Users/jaeheon/Desktop/Projects/project_surface-crack-detection/streamlit.ipynb
# 
# streamlit run /Users/jaeheon/Desktop/Projects/project_surface-crack-detection/streamlit.py
# 

# In[ ]:




