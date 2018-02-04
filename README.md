## blogXY
A machine learning production system to identify gender from bloggers' online activities.


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

3. Open a terminal, export environment variables and run airflow server
```
export FLASK_APP=./scripts/services/user.py;
export PYTHONPATH=.
export flask_shutdown_user=flask_admin
export flask_shutdown_password=flask_admin_stop_pass
```

4. Run airflow server
```
airflow webserver -p 8080
```

3. Open another terminal, export environment variables from previous the step and run scheduler
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

