#!/bin/bash
wget https://www.dropbox.com/s/1imf4oc1plaouw3/best_model.h5?dl=1 -O best_model.h5
python3 hw3_test.py $1 $2