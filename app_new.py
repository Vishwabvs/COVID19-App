import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_option('deprecation.showfileUploaderEncoding',False)

def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction

model = tf.keras.models.load_model('model.h5')

st.write("""
         # COVID19 Prediction
         """
         )

st.write("This is a simple image classification web app to predict COVID19")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("COVID19")
    elif np.argmax(prediction) == 1:
        st.write("Normal")
    else:
        st.write("Pneumonia")
             
              
    st.text("Probability (0: COVID19, 1: Normal, 2: Pneumonia)")
    st.write(prediction)