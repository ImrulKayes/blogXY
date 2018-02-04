# blogXY
A machine learning production system to identify gender from bloggers' online activity

1. Install required python packages
./scripts/bootstrap.sh

2. Open a terminal, export environment variables and run airflow server
export FLASK_APP=./scripts/services/user.py;
export PYTHONPATH=.
export flask_shutdown_user=flask_admin
export flask_shutdown_password=flask_admin_stop_pass
airflow webserver -p 8080

3. Open another terminal for aiflow scheduler, export environment variables from previous step and run scheduler
airflow scheduler

Examples:

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
