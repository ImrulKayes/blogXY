"""FLASK api service.
   The service provides the following functionalities,
   1. List of all users: e.g.,  curl http://localhost:5000/users
   2. List of friends of a user (SarahRubin): e.g., curl http://localhost:5000/friends/SarahRubin
   3. Prediction of the gender of a user (SarahRubin): curl http://localhost:5000/gender/SarahRubin
   4. Shutdown the server curl -u flask_admin:flask_admin_stop_pass http://localhost:5000/shutdown
   Shutdown requires authentication. We have to provide id (e.g., flask_admin) and password (e.g.,flask_admin_stop_pass).
   Those will be authenticated against saved id and password from configuration.
"""

from flask import Flask, request
from flask_restful import Resource, Api
from flask_httpauth import HTTPBasicAuth
from scripts.crawler.profile import get_blog_statistics
from scripts.model_generation import ModelGenerator
from sklearn.externals import joblib
import pandas as pd 
import logging
import conf.config as conf
from flask_restful.utils import cors
import functools
from werkzeug.exceptions import NotFound, ServiceUnavailable
import json
from scripts.services import make_json, crossdomain
import conf.config as conf

app = Flask(__name__)
auth = HTTPBasicAuth()

# Need authentication to shutdown the app
user_name = conf.flask_shutdown_user
user_pass = conf.flask_shutdown_password

# Set logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(conf.api_log_file)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Load user's friend data
def get_friends_data():
    user_friend_dic = {}
    for line in open(conf.friends_data, 'r'):
        vals = line.strip().split('|')
        user = vals[0]
        friends = vals[1:]
        user_friend_dic[user] = friends
    return user_friend_dic

def get_profile_data(username):
    # Get the user profile data
    profile_data = get_blog_statistics("{0}/{1}".format(conf.site, username), logger)  
    return  profile_data

# Get the saved gender ML model
def get_gender_model():
    clf = joblib.load(conf.model_output)
    return clf

# Load data and model when app starts
user_friend_data = get_friends_data()
model = get_gender_model()

# Handle shutdown, authentication and end point
@auth.verify_password
def verify_password(username, password):
    if not(username==user_name and password==user_pass):
        return False
    return True

@app.route('/shutdown')
@auth.login_required
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    logger.info("Shutting down server")
    return 'Server shutting down...'


@app.route("/", methods=['GET'])
def get_uris():
    return make_json({
        "uri": "/",
        "subresource_uris": {
            "users": "/users",
            "friends": "/friends/<username>",
            "gender": "/gender/<username>",
        }
    })

@app.route("/users", methods=['GET'])
def users_list():
    return make_json(user_friend_data.keys())

@app.route("/friends/<username>", methods=['GET'])
@crossdomain(origin='*')
def get_friends(username):
    if not user_friend_data.has_key(username):
        raise NotFound

    return make_json(user_friend_data[username])

@app.route("/profile/<username>", methods=['GET'])
@crossdomain(origin='*')
def get_profile(username):
    # Get user profile data
    profile_data = get_profile_data(username)
    if not profile_data:
        return make_json({'gender': 'Blogger no available. Request a registered blogger'})

    profile_schema = conf.blogster_profile_schema[1:]
    return make_json({profile_schema[i]:profile_data[i] for i in range(len(profile_schema))})

@app.route("/gender/<username>", methods=['GET'])
@crossdomain(origin='*')
def get_gender(username):
    # Get user profile data
    profile_data = get_profile_data(username)
    
    if not profile_data:
        return make_json({'gender': 'Blogger no available. Request a registered blogger'})

    profile_schema = conf.blogster_profile_schema[1:]

    # Create a dataframe from profile data
    df = pd.DataFrame([{profile_schema[i]:profile_data[i] for i in range(len(profile_schema))}])
    X = df[conf.initial_features]
    
    try:    
        # Generate input features
        model_generator = ModelGenerator()
        feature_pipeline = model_generator.create_feature_pipeline()
        feature_pipeline.fit(X)
        X = feature_pipeline.transform(X)
    except Exceptions:
        logger.info("Received request to predict user {0}, features {1}, got error in fit and transform".format(username, X))
        return make_json({'gender': 'Something went wrong!'})

    # Predict geneder
    predicted_gender = 'Male' if model.predict(X)[-1] == 0 else 'Female'
    logger.info("Received request to predict user {0}, features {1}, predicted {2}".format(username, X, predicted_gender))
        
    return make_json({'gender': predicted_gender})


if __name__ == '__main__':
    """export FLASK_APP=api.py, flask run
       example: curl http://localhost:5000/
                curl http://localhost:5000/users
                curl http://localhost:5000/friends/SarahRubin
                curl http://localhost:5000/gender/SarahRubin
                curl -u flask_admin:flask_admin_stop_pass http://localhost:5000/shutdown

    """
    app.run(debug=True)


