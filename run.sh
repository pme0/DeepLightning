#!/bin/bash

while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      mode="$2"
      shift # past argument
      shift # past value
      #echo mode=${mode}
      ;;
    --config)
      config="$2"
      shift # past argument
      shift # past value
      #echo config=${config}
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done


mlflow run . -e ${mode} -P cfg=${config}


#if [ "$#" -eq "0" ]; then
#    mlflow run . 
#else
#    mlflow run . -P cfg=$1
#fi

