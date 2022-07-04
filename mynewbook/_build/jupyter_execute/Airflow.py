#!/usr/bin/env python
# coding: utf-8

# # Apache Airflow

# 1. Installieren und Inzialisieren von Airflow nach Tutorial in der Vorlesung.
# 2. Erstellen einer DAG-Datei
# 3. Erstellen einzelnen Schritte

# In unserem Fall haben wir Apache Airflow dazu genutzt, das Abspeichern von neuen Bildern zur Verbesserung unseres Modells, zu Automatisieren.

# ## DAG-Datei

# In unserem Fall läuft das Funktion täglich einmal. Optimal wäre es wenn die Funktion immer dann ausgeführt wird, wenn unser Modell zu oft schlecht ausfällt

# In[1]:


#---------------------------------------
# SETUP

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

#---------------------------------------


# DEFAULT DAG ARGUMENTS

with DAG(
    'my_dag', 
    default_args={
        'depends_on_past': False,
        'email': ['my-email@example.com'],
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description='A simple DAG for Coin Model',
    schedule_interval= '@daily',
    start_date=datetime(2022, 6, 23),
    catchup=False,

) as dag:

    #---------------------------------------

    # DEFINE OPERATERS

    #BashOperator for image_to_file
    t1 = BashOperator(
        depends_on_past=False,
        task_id='image_to_file',
        bash_command="python '/Users/Kim/Library/Mobile Documents/com~apple~CloudDocs/UNI/HdM/Semester6/DataScience&MLOps/airflow/dags/image_to_file.py'",
    )

   #----------------

    # SETTING UP DEPENDENCIES
    t1


# ## Funktion zum Speichern der Bilder

# In[ ]:


#---------------------------------------
# SETUP

import glob
from PIL import Image
import random
import os

#---------------------------------------

path_to_new_extendet_dataset = '/Users/Kim/Library/Mobile Documents/com~apple~CloudDocs/UNI/HdM/Semester6/DataScience&MLOps/Coins/new_extended_dataset/original'

for filename in glob.glob("/Users/Kim/Library/Mobile Documents/com~apple~CloudDocs/UNI/HdM/Semester6/DataScience&MLOps/airflow/dags/*.jpg"):
    filename_splited = filename.split('/')
    filename_split = filename_splited[-1].split('.')
    for foldername in glob.glob(path_to_new_extendet_dataset+'/*'):
        foldername_split = foldername.split('/')
        if filename_split[0] == foldername_split[-1]:
            random_number = str(random.randint(0,10000))
            img = Image.open(filename)
            class_modell = str(filename_splited[-1])
            img = img.save(foldername+'/'+random_number+'_'+class_modell)
    os.remove(filename)

