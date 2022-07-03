
#imports for the Side
import streamlit as slit
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.preprocessing.image
from keras.preprocessing.image import img_to_array
import numpy as np
import io
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt

#load image
def load_image():
    uploaded_file = slit.file_uploader(label='Pick an image')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        slit.image(image_data)
        #convert image data to Bytes
        image_stream = io.BytesIO(image_data)
        return image_stream 
    else:
        return None


#load model
def load_model():
    model_path = '/Users/Kim/Library/Mobile Documents/com~apple~CloudDocs/UNI/HdM/Semester6/DataScience&MLOps/Coins/Models/Tuned/tuned_model'
    model = keras.models.load_model(model_path)
    return model

#get precition of one Coin
def predict_one_Coin (model, image):
    #resize image and convert to array
    img = Image.open(image)
    test_image = img.resize((200,200))
    input_arr = tf.keras.preprocessing.image.img_to_array(test_image)

    test_image = np.expand_dims(input_arr, axis=0)

    #predict image based on model
    prediction = model.predict(test_image)
    class_names=["1c", "2c", "5c", "10c", "20c", "50c", "1e", "2e"]

    #print out prediction
    slit.write(f"{class_names[np.argmax(prediction)]} with a { (100 * np.max(prediction)).round(2) } % prediction.")
    
    #return precition class to save image
    file_name = str(class_names[np.argmax(prediction)])
    return file_name


#layout streamlit page
def main():
    slit.set_page_config(page_title="One Coin")
    slit.title('One Coin')
    slit.title('Image upload')
    #load image
    image = load_image()
    #load model
    model = load_model()
    if image != None:
        slit.write('Calculating results...')
        #get prediction
        predicted = predict_one_Coin(model, image)
        result = slit.button('Press when Prediction was right')
        #save image to airflow
        if result:
            img = Image.open(image)
            img.save('/Users/Kim/Library/Mobile Documents/com~apple~CloudDocs/UNI/HdM/Semester6/DataScience&MLOps/airflow/dags/'+predicted+'.jpg')
            slit.success('File Saved')

if __name__ == '__main__':
    main()