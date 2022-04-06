#!/bin/bash

if [ "$#" -eq "0" ]; then
    mlflow run . 
else
    mlflow run . -P config=$1
fi