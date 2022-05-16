from sklearn import preprocessing
import streamlit as slit
import tensorflow as tf
from tensorflow import keras
import numpy as np

#model path
model_path = 'Model/maxConv2D_512_GlobalAveragePooling2D_50'

class_names = ['10c', '1c', '1e', '20c', '2c',
               '2e', '50c', '5c']

def load_image():
    uploaded_file = slit.file_uploader(label='Pick an image')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        slit.image(image_data)


#load model
def load_model():
    model = keras.models.load_model(model_path)
    return model

#predict to uploaded picture
def predict (model, categories, image):
    preprocessing = tf.data.Dataset.from_generator(
        lambda: image,
        output_types = (tf.float32, tf.uint8),
        output_shapes = ([None, 200, 200, 3], [None, 8]))

    prediction = model.predict(preprocessing)
    scores = tf.nn.softmax(prediction[0])
    scores = scores.numpy()
    results = {
          '1c': 0,
          '1e': 0,
          '10c': 0, 
          '2c': 0, 
          '20c': 0,
          '2e': 0,
          '50c': 0,
          '5c': 0,
    }
    result = f"{categories[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
    return result

#image upload
def main():
    slit.title('Image upload')
    image = load_image()
    model = load_model()
    result = slit.button('Run on image')
    if result:
        slit.write('Calculating results...')
        predict(model, class_names, image)

if __name__ == '__main__':
    main()

