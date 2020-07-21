# Disaster_Response

<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2020/01/pipe.png">

This repository contains a system analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.  
This project include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Motivation

The text analysis is a very striking technique because a lot of information that exists in almost all the datasets are flat texts and these techniques give us the possibility of obtaining very important data that are embedded in these flat texts. For example, the analysis of feelings.


### Instructions:

1. Install Requirements

pip install -r requirements.txt

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to localhost:5000/


### Structure of the project

```bash
|   .gitignore
│   README.md
│   requirements.txt
│
├───app
│   │   run.py
│   │
│   └───templates
│           go.html
│           master.html
│
├───data
│       DisasterResponse.db
│       disaster_categories.csv
│       disaster_messages.csv
│       process_data.py
│
└───models
        classifier.pkl
        train_classifier.py

```
