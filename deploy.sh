#!/bin/bash

cd api

artifact_path=$1

echo "Loading artifacts from '$artifact_path'."

taskline=$(grep 'task:' ${artifact_path}/cfg.yaml)
task=${taskline#*: }

python3 ${task}_app.py --artifact_path ${artifact_path}
