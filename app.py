import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
import os



# load the model
model = load_model(os.path.join('models','snakeclassification_model.h5'))


st.write('## Classifacation Snake model :snake:')
st.write("Ce projet s'inscrit dans un contexte de deep learning. L'objectif est a partir d'un model crée que celui-ci soit capable de reconnaitre a quel calsse apartien un serpent ")

img_extension = ['jpeg','jpg','png']
image_file = st.file_uploader('upload your image here ',type=img_extension)
  
    
#The function `resize_image` resized image with dimensions of 256x256 pixels.   
def resize_image(img): 
    img = cv2.imread(img)
    return tf.image.resize(img, (256,256))


def model_snake():
    """
    The function `model_snake` resizes an image, normalizes it, and then uses a model to predict the
    output.
    :return: the prediction made by the model on the resized image.
    """
    class_names = ['Boomslang', 'Cobra', 'Crotale', 'Taïpan', 'Vipère']
    img = resize_image(temp_file_path)
    pred = model.predict(np.expand_dims(img/255, 0))
    
    st.write(pred)
    predicted_class_index = np.argmax(pred)
    predicted_class_name = class_names[predicted_class_index]
    return st.write(f'Predicted Class: {predicted_class_name} with : {np.max(pred)}')
     


# filter to get only images files
if image_file is not None :
    
    st.image(image_file)
    # Save the uploaded file to a temporary location
    temp_file_path = "temp_image.jpg"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(image_file.getvalue())
    
    
    st.button('Predict', on_click=model_snake)


