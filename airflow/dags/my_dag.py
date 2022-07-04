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