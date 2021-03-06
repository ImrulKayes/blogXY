## blogXY
A machine learning production system to identify gender from bloggers' online activities.

* Collection of data using a crawler which uses Breadth first search (BFS).
* Feature engineering, hyperparameter tuning, model generation and saving using scikit-learn.
* Flask api service for handling requsts.
* Aiflow scheduler for collection of data, update ML models and parameters, and running the Flask api service. This update happens daily.

## Getting Started

1. Clone/download the repository (assuming you haven't already)
```
git clone https://github.com/ImrulKayes/blogXY.git
```

2. Move into the working directory with cd then install required python packages by running bootstrap script
```
cd blogXY
./scripts/bootstrap.sh
```

3. Open a terminal, go export environment variables and run airflow server
```
cd blogXY
export FLASK_APP=$PWD/scripts/flask_api/user.py;
export PYTHONPATH=$PWD
export flask_shutdown_user=flask_admin
export flask_shutdown_password=flask_admin_stop_pass
```

4. Run airflow server
```
airflow webserver -p 8080
```

3. Open another terminal, export environment variables from step 3 and run scheduler
```
airflow scheduler
```

Examples:

```
curl http://localhost:5000/
{
    "subresource_uris": {
        "friends": "/friends/<username>", 
        "gender": "/gender/<username>", 
        "users": "/users"
    }, 
    "uri": "/"
}

curl http://localhost:5000/users
    "edikacer", 
    "messyraindrop", 
    "eastbenchbears", 
    "sukumarjena"

curl http://localhost:5000/friends/eastbenchbears
[
    "artrex", 
    "christalon", 
    "earthkirk", 
    "iluvmater123", 
    "nickthink", 
    "whereabouts"
]

curl http://localhost:5000/gender/SarahRubin
{
    "gender": "Female"
}


curl -u flask_admin:flask_admin_stop_pass http://localhost:5000/shutdown

Server shutting down...
```

