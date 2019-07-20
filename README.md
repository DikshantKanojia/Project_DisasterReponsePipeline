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
       - templates contains html file for the web application


## 3. Results

    - An ETL pipleline was built to read data from two csv files, clean data, and save data into a SQLite database.
    - A machine learning pipepline was developed to train a classifier to performs multi-output classification on the 36 categories in the dataset.
    - A Flask app was created to show data visualization and classify the message that user enters on the web page.


## 4. Instructions


    - Run the following commands in the project's root directory to set up your database and model.
        - To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        - To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

    - Run the following command in the app's directory to run your web app. python run.py

    - Go to http://0.0.0.0:3001/

<a name="authors"></a>
## Authors

* [Dikshant Kanojia](https://github.com/DikshantKanojia)


## Acknowledgements

* [Udacity](https://www.udacity.com/) for providing such a complete Data Science Nanodegree Program
* [Figure Eight](https://www.figure-eight.com/) for providing messages dataset to train my model

<a name="screenshots"></a>
## Screenshots

1. This is an example of a message you can type to test Machine Learning model performance

![Sample Input](https://github.com/DikshantKanojia/Project_DisasterReponsePipeline/blob/master/app/Screen%20Shot%202019-07-19%20at%206.19.22%20PM.png)

2. After clicking **Classify Message**, you can see the categories which the message belongs to highlighted in green

![Sample Output](https://github.com/DikshantKanojia/Project_DisasterReponsePipeline/blob/master/app/disaster-response-project3.png)

3. The main page shows some graphs about training dataset, provided by Figure Eight

![Main Page](https://github.com/DikshantKanojia/Project_DisasterReponsePipeline/blob/master/app/disaster-response-project1.png)
