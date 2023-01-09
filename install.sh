#!/bin/bash

# clone repository
git clone https://github.com/pme0/DeepLightning
cd DeepLightning

# upgrade pip
pip3 install --upgrade pip

# install required packages
PACKAGES="mlflow"  # packages to be installed as space-separated string: "pkg1 pkg2 pkg3"
for package in $PACKAGES
do
    line=$(grep '$package' conda_env.yaml)
    version=${line#*==}
    pip3 install $package==$version
done
