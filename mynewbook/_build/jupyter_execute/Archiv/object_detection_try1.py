#!/usr/bin/env python
# coding: utf-8

# # Object Detection try 1
# 
# > Quelle: https://pyimagesearch.com/2020/06/22/turning-any-cnn-image-classifier-into-an-object-detector-with-keras-tensorflow-and-opencv/#pyis-cta-modal

# In[1]:


#Für Colab
#!git clone -b master https://github.com/HennFarr/Coins.git
#!pip install keras_tuner
#!pip install shap


# In[2]:


import tensorflow as tf
import tensorflow_datasets as tfds
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras


from keras import layers
import keras_tuner as kt
from keras.preprocessing.image import ImageDataGenerator, img_to_array,load_img
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import shap

import imutils
import argparse
import time
import cv2
from keras.applications.resnet import preprocess_input
from keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression


# In[3]:


local_model_dir = "Models/Tuned/tuned_model"
colab_model_dir = "/content/Coins/Models/Tuned/tuned_model"
model = keras.models.load_model(local_model_dir)


# In[4]:


def sliding_window(image, step, ws):
	# slide a window across the image
	for y in range(0, image.shape[0] - ws[1], step):
		for x in range(0, image.shape[1] - ws[0], step):
			# yield the current window
			yield (x, y, image[y:y + ws[1], x:x + ws[0]])


# In[5]:


def image_pyramid(image, scale, minSize):
	# yield the original image
	yield image
	# keep looping over the image pyramid
	while True:
		# compute the dimensions of the next image in the pyramid
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# yield the next image in the pyramid
		yield image


# In[6]:


# initialize variables used for the object detection procedure
WIDTH = 500
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = (50,50) #eval(args["size"])
INPUT_SIZE = (200, 200)


# In[7]:


local_img="real_test_data/multi/IMG_20220620_105307.jpg"
colab_img="/content/Coins/real_test_data/multi/IMG_20220620_105307.jpg"


# In[8]:


# load the input image from disk, resize it such that it has the
# has the supplied width, and then grab its dimensions
orig = cv2.imread(local_img) #args["image"]
orig = imutils.resize(orig, width=WIDTH)
(H, W) = orig.shape[:2]


# In[9]:


# initialize the image pyramid
pyramid = image_pyramid(orig, scale=PYR_SCALE, minSize=ROI_SIZE)
# initialize two lists, one to hold the ROIs generated from the image
# pyramid and sliding window, and another list used to store the
# (x, y)-coordinates of where the ROI was in the original image
rois = []
locs = []
# time how long it takes to loop over the image pyramid layers and
# sliding window locations
start = time.time()


# In[10]:


# loop over the image pyramid
for image in pyramid:
	# determine the scale factor between the *original* image
	# dimensions and the *current* layer of the pyramid
	scale = W / float(image.shape[1])
	# for each layer of the image pyramid, loop over the sliding
	# window locations
	for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
		# scale the (x, y)-coordinates of the ROI with respect to the
		# *original* image dimensions
		x = int(x * scale)
		y = int(y * scale)
		w = int(ROI_SIZE[0] * scale)
		h = int(ROI_SIZE[1] * scale)
		# take the ROI and preprocess it so we can later classify
		# the region using Keras/TensorFlow
		roi = cv2.resize(roiOrig, INPUT_SIZE)
		roi = img_to_array(roi)
		roi = preprocess_input(roi)
		# update our list of ROIs and associated coordinates
		rois.append(roi)
		locs.append((x, y, x + w, y + h))
		# check to see if we are visualizing each of the sliding
		# windows in the image pyramid
		if 0 > 0:			#args["visualize"] # Dauert ewig
			# clone the original image and then draw a bounding box
			# surrounding the current region
			clone = orig.copy()
			cv2.rectangle(clone, (x, y), (x + w, y + h),
				(0, 255, 0), 2)
			# show the visualization and current ROI
			cv2.imshow("Visualization", clone)
			cv2.imshow("ROI", roiOrig)
			cv2.waitKey(0)		


# In[11]:


# show how long it took to loop over the image pyramid layers and
# sliding window locations
end = time.time()
print("[INFO] looping over pyramid/windows took {:.5f} seconds".format(
	end - start))
# convert the ROIs to a NumPy array
rois = np.array(rois, dtype="float32")
# classify each of the proposal ROIs using ResNet and then show how
# long the classifications took
print("[INFO] classifying ROIs...")
start = time.time()
preds = model.predict(rois)
end = time.time()
print("[INFO] classifying ROIs took {:.5f} seconds".format(
	end - start))
# decode the predictions and initialize a dictionary which maps class
# labels (keys) to any ROIs associated with that label (values)
#preds = imagenet_utils.decode_predictions(preds, top=1)
labels = {}


# In[12]:


class_names=["1c", "2c", "5c", "10c", "20c", "50c", "1e", "2e"]
preds2=[]
for i in preds:
    preds2.append([np.argmax(i), class_names[np.argmax(i)],list(i)])


# In[13]:


preds2[0:20]


# In[14]:


# loop over the predictions
for (i,p) in enumerate(preds2):
	# grab the prediction information for the current ROI
	(coinID, label, prob) = p
	# filter out weak detections by ensuring the predicted probability
	# is greater than the minimum probability
	if 0.85 >= prob[coinID] >= 0.65:	#args["min_conf"]
		# grab the bounding box associated with the prediction and
		# convert the coordinates
		box = locs[i]
		# grab the list of predictions for the label and add the
		# bounding box and probability to the list
		L = labels.get(label, [])
		L.append((box, prob[coinID]))
		labels[label] = L


# In[15]:


labels


# In[16]:


# loop over the labels for each of detected objects in the image
for label in labels.keys():
	# clone the original image so that we can draw on it
	print("[INFO] showing results for '{}'".format(label))
	clone = orig.copy()
	# loop over all bounding boxes for the current label
	for (box, prob) in labels[label]:
		# draw the bounding box on the image
		(startX, startY, endX, endY) = box
		cv2.rectangle(clone, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
	# show the results *before* applying non-maxima suppression, then
	# clone the image again so we can display the results *after*
	# applying non-maxima suppression
	cv2.imshow("Before", clone)
	clone = orig.copy()
    # extract the bounding boxes and associated prediction
	# probabilities, then apply non-maxima suppression
	boxes = np.array([p[0] for p in labels[label]])
	proba = np.array([p[1] for p in labels[label]])
	boxes = non_max_suppression(boxes, proba)
	# loop over all bounding boxes that were kept after applying
	# non-maxima suppression
	for (startX, startY, endX, endY) in boxes:
		# draw the bounding box and label on the image
		cv2.rectangle(clone, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.putText(clone, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
	# show the output after apply non-maxima suppression
	cv2.imshow("After", clone)
	cv2.waitKey(0)

