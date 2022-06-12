#!/usr/bin/env python
# coding: utf-8

# # Data Exploration

# In[1]:


from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np


# ## Datensatze
# ### "Extended"

# In[2]:


#rescale nötig für graphische Darstellung
datagen=ImageDataGenerator(rescale=1./255)   
data_dir = 'new_extended_dataset/original'

extended_data = datagen.flow_from_directory(data_dir,
    class_mode="categorical", 
    subset="training")


# Der Datensatz mit dem wir unser finales Model trainieren besteht aus 9664 Bildern die in acht unterschiedliche Klassen unterteilt sind:
# - 1 cent
# - 2 cent
# - 5 cent
# - 10 cent
# - 20 cent
# - 50 cent
# - 1 Euro
# - 2 Euro 
# 
# Dieser Datensatz ist aber aus zwei verschiedenen Datensätzen zusammengesetzt, die wir während unserer Recherche Gefunden haben. Der Unterschied liegt darin, dass die Aufnahmen der Datensätze qualitativ unterschiedlich sind. Der Größer Datensatz besteht aus hochauflösenden Bildern von Münzen, die alle aus dem selben Winkel und auf dem selben Untergrund fotografiert wurden. Hinzukommt, dass alle Münzen in verschiedenen Rotationen vorliegen. Jede Münze weißt zwischen 12-41 unterschiedliche Rotationen auf. Der zweite Datensatz Besteht aus etwas weniger hochauflösenden Bilder. Desweiteren sind die Münzen aus verschiedenen Winkeln fotografiert und an den Rändern sind teilweise andere Münzen zu sehen, die die Münze im Mittelpunkt auch teils überlappen.
# 
# Wichtig Anzumerken ist, dass alle Bilder auf die "Zahl"-Seite der Euro-Münzen abbilden. Wir haben uns aktiv gegen eine Klassifikation beider Seiten entschieden, da sich diese von Land zu Land unterscheiden und je nach Datensatz verschiedene Ländermünzen mehr oder weniger vertreten sind.

# ### "Smal"
# !Erinnerung: test und train zusammen führen!
# 
# Für den "Extended"-Datensatz wurden die Bilder auf denen die "Kopf"-Seite der Münze zu sehen war entfernt.

# In[3]:


small_data = datagen.flow_from_directory("small data set/classified/train",
    class_mode="categorical", 
    subset="training")


# In[4]:


small_data_images, small_data_labels = small_data.next()

fig1 = plt.figure(figsize=(20,10))
fig1.subplots_adjust(wspace=0.2, hspace=0.4)

# Lets show the first 15 images
for i, img in enumerate(small_data_images[:15]):
    ax = fig1.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(img)
    plt.title(np.argmax(small_data_labels[i]))


# ### "Big"

# In[5]:


big_data = datagen.flow_from_directory("big data set/original",
    class_mode="categorical", 
    subset="training")


# In[6]:


big_data_images, big_data_labels = big_data.next()

fig1 = plt.figure(figsize=(20,10))
fig1.subplots_adjust(wspace=0.2, hspace=0.4)

# Lets show the first 15 images
for i, img in enumerate(big_data_images[:15]):
    ax = fig1.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(img)
    plt.title(np.argmax(big_data_labels[i]))


# In[7]:


extended_data_images, extended_data_labels = extended_data.next()
print(extended_data_images.shape, extended_data_images.dtype)
print(extended_data_labels.shape, extended_data_labels.dtype)


# In[8]:


fig1 = plt.figure(figsize=(20,10))
fig1.subplots_adjust(wspace=0.2, hspace=0.4)

# Lets show the first 15 images
for i, img in enumerate(extended_data_images[:15]):
    ax = fig1.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(img)
    plt.title(np.argmax(extended_data_labels[i]))

