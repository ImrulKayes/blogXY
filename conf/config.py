import os
import numpy as np
from datetime import datetime, timedelta

home = os.environ['PYTHONPATH']

thread_num = 20
site = "http://www.blogster.com"
seed_blogger_file = "{0}/data/seed.txt".format(home)
profile_data_file = "{0}/data/profile_data.txt".format(home)
edge_data_file = "{0}/data/adj_list.txt".format(home)

log_file = '{0}/data/crawl_log.log'.format(home)
crawler_log_file = '{0}/logs/crawl_log.log'.format(home)
model_generation_log_file = '{0}/logs/model_generation_log.log'.format(home)
api_log_file = '{0}/logs/api_log.log'.format(home)

profile_data = '{0}/data/profile_data.txt'.format(home)
friends_data = '{0}/data/adj_list.txt'.format(home)

model_output = '{0}/model/gender_model.pkl'.format(home)

airflow_email = 'airflow_blogster@gmail.com'
airflow_start_date = datetime(2017, 11, 3),
airflow_retries = 0
airflow_retry_delay = 5
airflow_queue = "general"

flask_shutdown_user = 'flask_admin'
flask_shutdown_password = 'flask_admin_stop_pass'


test_size=0.25
blogster_profile_schema = [
    'serial', 
    'name',
    'visibility',
    'birthday',
    'gender',
    'location',
    'status',
    'joined',
    'job',
    'language',
    'blogTraffic',
    'posts',
    'myComments',
    'userComments',
    'photos',
    'friends',
    'following',
    'followers',
    'points',
    'lastOnline']
blogster_profile_schema_type ={
    'serial': np.int32, 
    'name': object,
    'visibility': object,
    'birthday': object,
    'gender': object,
    'location': object,
    'status': object,
    'joined': object,
    'job': object,
    'language': object,
    'blogTraffic': np.float64,
    'posts': np.float64,
    'myComments': np.float64,
    'userComments': np.float64,
    'photos': np.float64,
    'friends': np.float64,
    'following': np.float64,
    'followers': np.float64,
    'points': np.float64,
    'lastOnline': object
}

initial_features = [
    'birthday', 
    'location', 
    'status', 
    'job', 
    'language',
    'blogTraffic', 
    'posts', 
    'myComments', 
    'userComments',
    'points', 
    'lastOnline',
    'friends',
    'following',
    'followers',
    'photos']

binary_features = [
    'birthday', 
    'location', 
    'status', 
    'job', 
    'language']

general_features = [ 
    'blogTraffic', 
    'posts', 
    'myComments', 
    'userComments',
    'points', 
    'friends',
    'following',
    'followers',
    'photos']




