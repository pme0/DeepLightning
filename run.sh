#!/bin/bash

mlflow run . -e $1 -P cfg=$2


#while [[ $# -gt 0 ]]; do
#  case $1 in
#    --mode)
#      mode="$2"
#      shift # past argument
#      shift # past value
#      #echo mode=${mode}
#      ;;
#    --config)
#      config="$2"
#      shift # past argument
#      shift # past value
#      #echo config=${config}
#      ;;
#    -*|--*)
#      echo "Unknown option $1"
#      exit 1
#      ;;
#  esac
#done


#if [ "$#" -eq "0" ]; then
#    mlflow run . 
#else
#    mlflow run . -P cfg=$1
#fi

