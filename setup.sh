#!/bin/sh
cd ~/CORN
mkdir outputs
sudo apt install python3-pip
sudo apt-get install libsuitesparse-dev
pip install -r requirements.txt
chmod +x run.sh

