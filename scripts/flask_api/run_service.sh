#!/bin/bash
# Check if there is a screen session for the flask api service 
screen -ls| grep run_flask;

# If running screen session and service is running then stop the service and screen 
if  [ $? -eq 0 ]; then  
    netstat -anp tcp | grep 127.0.0.1.5000; 
    if  [ $? -eq 0 ]; then  
        curl -u $flask_shutdown_user:$flask_shutdown_password http://localhost:5000/shutdown;
    fi
    screen -X -S run_flask kill;
fi

# Run the flask api service in a screen
screen -S run_flask -d -m  bash -c "cd; flask run"
echo "Started the service"
