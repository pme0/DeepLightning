#!/bin/bash

if [ "$#" -eq "0" ]; then
    mlflow run . 
else
    mlflow run . -P cfg=$1
fi