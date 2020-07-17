# Disaster-Response-Pipeline
### Introduction:
How first reponders respond in time of distress is a very important part of disaster response and so this project was created to help them classify thousands of messages they recieve to enable them respond as fast as possible.

This project required building an ETL pipeline and then training the data to classify future messages. Flask library was used to build the web app interface (template)

### Folders and Files used:
**app** : This contains the folder **template** which houses the main page of the web app and the classification result page of web app. **run.py** is the flask file that runs app.

**data** : Contains the data used for this project and the python script for Extracting Transforming and Loading (ETL) the data, **process_data.py**

**models** : This contains an machine learning pipeline, **train_classifier** and the save model to classify messages, **classifier.pkl**

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
