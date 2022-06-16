import streamlit as slit
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.preprocessing.image
import numpy as np
import os
import io
from PIL import Image
from torchvision import transforms

#bildhochladen vs. webcam ansteuern!!
#https://docs.streamlit.io/library/api-reference/widgets/st.camera_input
#https://docs.streamlit.io/library/api-reference/widgets/st.camera_input#tensorflow
def load_image():
    uploaded_file = slit.file_uploader(label='Pick an image')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        slit.image(image_data)
        return Image.open(io.BytesIO(image_data))
        
    else:
        return None


#load model
def load_model():
    model_path = '/Users/Kim/Library/Mobile Documents/com~apple~CloudDocs/UNI/HdM/Semester6/DataScience&MLOps/Coins/Models/Tuned/tuned_model'
    model = keras.models.load_model(model_path)
    return model

#predict to uploaded picture tensorflow komponenten
def predict (model, image):
    # preprocess = transforms.Compose([
    #     transforms.Resize(200,200),])
    # preprocessed_image = preprocess(image)
    test_image = image.resize((200,200))
    input_arr = tf.keras.preprocessing.image.img_to_array(test_image)

    test_image = np.expand_dims(input_arr, axis=0)

    prediction = model.predict(test_image)
    class_names=["1c", "2c", "5c", "10c", "20c", "50c", "1e", "2e"]

    result = f"{class_names[np.argmax(prediction)]} with a { (100 * np.max(prediction)).round(2) } % prediction." 
    return result

#image upload
def main():
    slit.title('Image upload')
    image = load_image()
    # slit.write(image)
    img = '/Users/Kim/Library/Mobile Documents/com~apple~CloudDocs/UNI/HdM/Semester6/DataScience&MLOps/Coins/new_extended_dataset/original/1c/IMG_20190707_010415_4.jpg'
    model = load_model()
    result = slit.button('Run on image')
    if result:
        slit.write('Calculating results...')
        predicted = predict(model, image)
        slit.write(predicted)

if __name__ == '__main__':
    main()

