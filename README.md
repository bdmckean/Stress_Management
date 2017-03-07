# Stress_Management
Data Science: Build and deploy a classifier to label uploaded sessions as bad or good

# Brian McKean

## Overview

[Biotrak Health](https://www.biotrakhealth.com "BioTrak Health Homepage") is a company making a headband and app that can be used to manage stress via biofeedback. BioTrak is in the early stages of developing its technology. The first headbands have been developed and are being tested in a trial with friends and family of the the company members. The purpose of the classifier is to classify each session as an effective use or not effective use of the system.

![Alt text](./images/BioTrakLogo.png)

Once these sessions have been classified we can begin to look for patterns and root causes.

With root causes we can start to look for ways to eliminated problems and improve the product.

## Introduction

The system architecture currently includes a custom headband with BioTrak Health developed hardware and software, an iOS app, and a cloud back end with a REST API for use by the app.  

### Headband
The biofeedback device is a headband. The headband has sensors that detect electrical activity in forehead muscles. There is always activity present but actions like talking or chewing cause nearby muscles to be activated and this activity is detected by the headband.

![Alt text](./images/ManWearingHeadBand.png)

### App
When the user stops activity and tries to relax, the base tension level of the forehead muscles is detected. Throughout the process the activity level is communicated to a mobile app. The mobile app provides three functions:
1. Display of tension level to the user
2. A variety of coaching sessions to help with tension reduction
3. Communication with the cloud REST api to store and retrieve data, including upload of session data at the conclusion of the session.

![Alt text](./images/PhoneAndHeadBand.png)

### Cloud Back End
The cloud back end is built on AWS with a Node/Express/MongoDB stack running on Ubuntu. The app uploads each session into the mongoDB session collection via a REST interface

![Alt text](./images/Architecture.png)

##  Project Goal

The project goal is to develop a machine learning model to classify each uploaded session as either an 'effective' or 'not_effective' use of the product. Since the goal is to find bad sessions the model will use a binary classifier with True='bad session' and False='Good Session'

### Session information
Each session uploaded includes a record of the data sent from the headband to the app in addition to metadata generated by the app. The app generated metadata includes user information, date and time, and session metadata such as which type of session.

This project uses the session data generated by the headband for classifying the session.
