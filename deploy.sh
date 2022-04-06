#!/bin/bash

cd api

if [ "$#" == "1" ]; then
    artifact="$HOME/data/checkpoints"
else
    artifact=$2
fi

echo "Loading artifacts from '$artifact'."

case "$1" in
  "classify") 
    python3 classify_app.py --artifact_path $artifact
    ;;
  "reconstruct")
    echo "ERROR: Not Implemented yet."
    #python3 reconstruct_app.py --artifact_path $artifact
    ;;
  *)
    echo "ERROR: Unknown task."
    ;;
esac


