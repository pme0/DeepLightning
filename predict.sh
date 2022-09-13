#!/bin/bash

# When redirected, curl shows progress meter:
#----------------------------------------------------------------------------------
#% Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
#                                 Dload  Upload   Total   Spent    Left  Speed
#100  1253  100    37  100  1216   3127   100k --:--:-- --:--:-- --:--:--  107k 
#----------------------------------------------------------------------------------
# To supress the progress meter without supressing errors, use:
# `curl --fail --silent --show-error [OTHERSTUFF] 2>/dev/null`
# This will send errors to STDERR.

if [ "$#" -eq "0" ]; then
    for x in {0..9}; do
        curl --fail --silent --show-error -X POST -F image=@media/digits/digit${x}.jpg http://localhost:5000/predict 2>/dev/null | python3 -m json.tool
    done
else
    curl --fail --silent --show-error -X POST -F image=@$1 http://localhost:5000/predict 2>/dev/null | python3 -m json.tool
fi

