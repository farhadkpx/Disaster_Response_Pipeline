## Disaster_Response_Pipeline Project
An ETL + ML + Python App for Udacity Data Science Nano Degree course

## Project portrayal
In this project, we will build a model corresponding with ML pipeline to classify real messages those were sent during natural disasters. These message can be categorized as medical help, aid related, search and rescue type.This upcoming ML processes are pipeline will helps us to categorize those messages in such a way that we can send messages to specific appropriate disaster relief agency.

This project will involve the building
- basic ETL that loads, categorizes, merges, cleans and Stores data in a SQLite database
- Machine Learning model pipeline builds a text processing then Trains and tunes using GridSearchCV
- Flask Web App visualizes classified messages result.

This is also a multi-label classification task, since a message can belong to one or more categories. We will be working with a data set provided by [Figure Eight](https://www.figure-eight.com/) containing real messages that were sent during disaster events.

In this project we'll be applying our  natural language processing,  machine learning and data engineering skills as a data scientist. Analyzing disaster data from Figure Eight to build a model for an API that classifies disaster messages. This project will show off your software skills, including your ability to create basic data pipelines and write clean, organized code!

Our project will include a Flask web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

Below are a few screenshots of the web app.
![Screenshot of Web App](webapp_screenshot.JPG)

## File Description
~~~~~~~
        disaster_response_pipeline
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data
                |-- disaster_message.csv
                |-- disaster_categories.csv
                |-- DisasterResponse.db
                |-- ETL Pipeline Preparation.ipynb
                |-- process_data.py
          |-- glove.6B
                |--- glove.6B.50d.txt
                |--- glove.6B.100d.txt
                |--- glove.6B.200d.txt
                |--- glove.6B.300d.txt
          |-- models
                |-- glove_vectorizer.py
                |-- ML Pipeline Preparation.ipynb
                |-- train_classifier.py
~~~~~~~
### Description of key files
1. run.py: Script to run the web app
2. disaster_message.csv: Contains the original disaster messages
3. disaster_categories.csv: Contains the labels of the disaster messages
4. process_data.py: Runs the ETL pipeline to process data from both disaster_message.csv and disaster_categories.csv and load them into an SQLite database, DisasterResponse.db.
5. train_classifier.py: Runs the ML pipeline to classify the messages. The pipeline will build the model, optimize it using grid search and print the model's evaluation. It will then save the classifier model.

## Pre-trained GloVe Embeddings (Optional)
If you want to use the MLP Neural Network model with pre-trained GloVe embeddings, you need to download the pre-trained embeddings from this [link](http://nlp.stanford.edu/data/glove.6B.zip). The downloaded files should be placed in the following folder structure:

~~~~~~~
        disaster_response_pipeline
          |-- app
          |-- data
          |-- glove.6B
               |--- glove.6B.50d.txt
               |--- glove.6B.100d.txt
               |--- glove.6B.200d.txt
               |--- glove.6B.300d.txt
          |-- models
~~~~~~~

## Instructions:
1. In the disaster_response_pipeline directory

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier (Adaboost with Tfidf Vectorizer) and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
    - To run ML pipeline that trains classifier (MLP with GloVe Embeddings) and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl --model_type 2`

2. In the app directory, run the following command to run the web app.
    `python run.py`

3. Go to http://localhost:3001 to view the web app

## Installations
Anaconda, Nltk, re, SQLAlchemy
