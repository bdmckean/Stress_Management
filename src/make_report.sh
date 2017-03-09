#!/bin/bash
cd report
/home/ubuntu/report/mongo_get.sh
python /home/ubuntu/report/BioTrak-Classify-Trial.py
python /home/ubuntu/report/mkdoc.py
python /home/ubuntu/report/mail2.py
