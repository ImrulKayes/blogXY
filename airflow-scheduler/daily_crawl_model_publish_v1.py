# External Imports
from airflow import DAG
from airflow.operators import BashOperator
from airflow.operators import PythonOperator
from airflow.utils import apply_defaults
from datetime import datetime, timedelta
import subprocess
import logging
import os, signal
import time

# Local import
from conf.config import airflow_start_date
import conf.config as conn

# Home directory
home_directory = conn.home

class Task(object):
    @staticmethod
    def add_run_cralwer(dag):
        BashOperator(
            task_id = 'run_cralwer',
            bash_command = 'python {0}/scripts/crawler/crawler.py'.format(home_directory),
            dag = dag)

    @staticmethod
    def add_run_model_generator(dag):
        BashOperator(
            task_id = 'run_model_generator',
            bash_command = 'python {0}/scripts/modeler/model_generator.py'.format(home_directory),
            dag = dag)
    
    @staticmethod
    def add_stop_start_flask_api(dag):
        BashOperator(
            task_id = 'stop_start_flask_api',
            bash_command = "sh {0}/scripts/flask_api/run_service.sh ".format(home_directory),
            dag = dag)


def make_dag(name, default_args):

    # Create the DAG
    dag = DAG(name, schedule_interval='@daily', default_args = default_args)

    # Instantiate tasks for the dag
    Task.add_run_cralwer(dag)
    Task.add_run_model_generator(dag)
    Task.add_stop_start_flask_api(dag)

    # Setup dependencies 
    dag.set_dependency('run_cralwer', 'run_model_generator')
    dag.set_dependency('run_model_generator', 'stop_start_flask_api')

    return dag

DAG_ARGS = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2018, 02, 19),
    'email': ['sample_email'],
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': conn.airflow_retries,
    'retry_delay': timedelta(minutes = conn.airflow_retry_delay),
}

dagMaster = make_dag('daily_crawl_model_publish', DAG_ARGS)