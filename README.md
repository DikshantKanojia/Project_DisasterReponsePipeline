# Project_DisasterReponsePipeline

## Project Motivation
In this project, I appled data engineering, natural language processing, and machine learning skills to analyze message data that people send during disasters to build a model for an API that classifies disaster messages. These messages could potentially be sent to appropriate disaster relief agencies.

## Table of contents

   1. Installation 
   2. File Descriptions
   3. Results
   4. Instructions


## 1. Installations
Beyond the Anaconda distribution of Python, the following packages need to be installed for nltk:

   - punkt
   - wordnet
   - stopwords


## 2. File Descriptions
There are three main foleders:

    1. data
       - disaster_categories.csv: dataset including all the categories
       - disaster_messages.csv: dataset including all the messages
       - process_data.py: ETL pipeline scripts to read, clean, and save data into a database
       - DisasterResponse.db: output of the ETL pipeline, i.e. SQLite database containing messages and categories data
    2. models
       - train_classifier.py: machine learning pipeline scripts to train and export a classifier
       - classifier.pkl: output of the machine learning pipeline, i.e. a trained classifer
    3. app
       - run.py: Flask file to run the web application
       - templates contains html file for the web applicatin


## 3. Results

## 4. Instructions

