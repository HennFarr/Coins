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

#bildhochladen vs. webcam ansteuern!!
#https://docs.streamlit.io/library/api-reference/widgets/st.camera_input
#https://docs.streamlit.io/library/api-reference/widgets/st.camera_input#tensorflow
def load_image():
    uploaded_file = slit.file_uploader(label='Pick an image')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        slit.image(image_data)
        image_stream = io.BytesIO(image_data)
        img = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
        
    else:
        return None


#load model
def load_model():
    model_path = '/Users/Kim/Library/Mobile Documents/com~apple~CloudDocs/UNI/HdM/Semester6/DataScience&MLOps/Coins/Models/Tuned/tuned_model'
    model = keras.models.load_model(model_path)
    return model

#predict to uploaded picture tensorflow komponenten
def predict_one_Coin (model, image):
    test_image = image.resize((200,200))
    input_arr = tf.keras.preprocessing.image.img_to_array(test_image)

    test_image = np.expand_dims(input_arr, axis=0)

    prediction = model.predict(test_image)
    class_names=["1c", "2c", "5c", "10c", "20c", "50c", "1e", "2e"]

    result = f"{class_names[np.argmax(prediction)]} with a { (100 * np.max(prediction)).round(2) } % prediction." 
    return result


def adjust_gamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

def findCoins(img, showCoins = False):
    scaling = 600.0/max(img.shape[0:2])
    img_gray = cv2.resize(img, None, fx=scaling, fy=scaling)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.blur(img_gray, (5,5))
    coins = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1.2, 30, param2 = 35, minRadius = 20, maxRadius = 50)
    coins = (np.round(coins[0,:]) / scaling).astype("int")
    return coins

#predict to uploaded picture tensorflow komponenten
def predict_more_Coin (model, image):
    coins = findCoins(image, showCoins = True)
    maxRadius = np.amax(coins,0)[2]+10

    class_names=["1c", "2c", "5c", "10c", "20c", "50c", "1e", "2e"]
    n=int(len(coins)/2)
    num_rows = n+2
    num_cols = n-2
    num_images = num_rows*num_cols

    clone = image.copy()

    for i,(x,y,r) in enumerate(coins):
        img_coin = image[y-maxRadius:y+maxRadius, x-maxRadius:x+maxRadius]
        if img_coin.shape[0]==0 or img_coin.shape[1]==0:
            continue

        img_coin = cv2.resize(img_coin, (200,200))
        img_x = adjust_gamma(img_coin, gamma=1)    #Gamma Experiment: Je nach Gamma-Wert Performed das Modell besser oder Schlechter. Stark vom Bild abhängig
        img_x = img_to_array(img_x)                    
        img_x = img_x.reshape((1,) + img_x.shape)

        s_prediction = model.predict(img_x)
        pred = f"{class_names[np.argmax(s_prediction)]} with { (100 * np.max(s_prediction)).round(2) } % acc"
        # slit.write(pred)

        cv2.rectangle(clone, (x-maxRadius, y-maxRadius), (x+maxRadius, y+maxRadius),(0,100,0), 5)
        cv2.rectangle(clone,(x-maxRadius, y-maxRadius),(x+maxRadius-20,y-maxRadius-50),(0,100,0),-1)
        cv2.putText(clone, pred, (x-maxRadius, y-maxRadius-10),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        

    plt.figure(figsize=(30,40))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(clone)
    plt.savefig('xyz.png',dpi = 100 ,transparent=True)
    slit.image('xyz.png')
    os.remove('xyz.png')

#image upload
def main():
    slit.title('Image upload')
    image = load_image()
    model = load_model()
    result = slit.button('Image with one Coin')
    if result:
        slit.write('Calculating results...')
        predicted = predict_one_Coin(model, image)
        slit.write(predicted)
    
    result_more_Coins = slit.button('Image with more Coins')
    if result_more_Coins:
        slit.write('Calculating results...')
        predicted = predict_more_Coin(model, image)

if __name__ == '__main__':
    main()

