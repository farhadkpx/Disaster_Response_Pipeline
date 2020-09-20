## Disaster_Response_Pipeline Project
An ETL + ML + Python App for Udacity Data Science Nano Degree course

## Project Description
In this project, we will build a model to classify messages that are sent during disasters. There are 36 pre-defined categories, and examples of these categories include Aid Related, Medical Help, Search And Rescue, etc. By classifying these messages, we can allow these messages to be sent to the appropriate disaster relief agency. This project will involve the building of a basic ETL and Machine Learning pipeline to facilitate the task. This is also a multi-label classification task, since a message can belong to one or more categories. We will be working with a data set provided by [Figure Eight](https://www.figure-eight.com/) containing real messages that were sent during disaster events.

Two types of models are available to classify the messages. 

    1. The first model type, which is also the default option, is an Adaboost Classifier utilizing Tfidf vectorizer to transform the    messages. 

    2. The second model is an MLP Neural Network utilizing pre-trained GloVe embeddings to transform the messages. It will take the simple average of word vectors to get the message vector.

Finally, this project contains a web app where you can input a message and get classification results.

![Screenshot of Web App](webapp_screenshot.JPG)
