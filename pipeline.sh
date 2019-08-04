#!/usr/bin/env bash

# Make sure you are in the repo folder in your system.
# Make sure you have replaced the dummy data from github to the actual task_data.csv
python3 -m venv ranking
source ranking/bin/activate
pip install -r requirements.txt

mkdir -p output

python main.py