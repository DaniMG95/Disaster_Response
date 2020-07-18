import sys
import sqlalchemy as db
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import numpy as np
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """Load dataset into database of filepath and separate in X and Y

    Parameters:
    database_filepath (string): path of database

    Returns:
    X:df
    Y:df
    Y.columns:list

    """
    engine = db.create_engine('sqlite:///'+database_filepath)
    df =pd.read_sql("SELECT * FROM disaster",con=engine)
    X = df["message"]
    Y = df.drop(["id","message","original","genre"],axis=1)
    return X,Y,Y.columns.tolist()


def tokenize(text):
    """Clean, tokenize and lemmatizer text

    Parameters:
    text (string): text which we want tokenize

    Returns:
    tokens:list

    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

def build_model():
    """Build Pipeline and create model

    Returns:
    cv:GridSearchCV

    """
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [30, 50, 80 ]
    }

    cv = GridSearchCV(model, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test):
    """Evaluate model get accuracym precision and recall of model

    Parameters:
    model (Model): Model which we want evaluate
    X_test (dataframe): Test
    Y_test (dataframe): label of Test


    """
    y_pred = model.predict(X_test)
    Y_test=np.array(Y_test)
    accuracy=accuracy_multioutput(Y_test,y_pred)
    precision=precision_multioutput(Y_test,y_pred)
    recall=recall_multioutput(Y_test,y_pred)
    print("Accuracy : {} , Precision : {} , Recall : {}".format(accuracy, precision,recall))


def accuracy_multioutput(y_true, y_pred):
    """Calculates multioutput classification average accuracy-score.

    Parameters:
    
    y_true: numpy.array Ground truth of the labels
    y_pred: numpy.array Predictions of the labels

    Returns:
    average_accuracy_score: Float Averaged accuracy-score of all the labels
    """

    score_accuracy_list = []
    for i in range(y_true.shape[1]):
        accuracy = accuracy_score(y_true[:,i],y_pred[:,i])
        score_accuracy_list.append(accuracy)

    average_accuracy_score = np.mean(score_accuracy_list)
    return average_accuracy_score


def precision_multioutput(y_true, y_pred):
    """Calculates multioutput classification average precision-score.

    Parameters:
    
    y_true: numpy.array Ground truth of the labels
    y_pred: numpy.array Predictions of the labels

    Returns:
    average_precision_score: Float Averaged precision-score of all the labels
    """

    score_precision_list = []
    for i in range(y_true.shape[1]):
        precision = precision_score(y_true[:,i],y_pred[:,i])
        score_precision_list.append(precision)

    average_precision_score = np.mean(score_precision_list)
    return average_precision_score


def recall_multioutput(y_true, y_pred):
    """Calculates multioutput classification average recall-score.

    Parameters:
    y_true: numpy.array Ground truth of the labels
    y_pred: numpy.array Predictions of the labels

    Returns:
    average_recall_score: Float Averaged recall-score of all the labels
    """

    score_recall_list = []
    for i in range(y_true.shape[1]):
        recall = recall_score(y_true[:,i],y_pred[:,i])
        score_recall_list.append(recall)

    average_recall_score = np.mean(score_recall_list)
    return average_recall_score




def save_model(model, model_filepath):
    """Save model into model_filepath

    Parameters:
    
    model: (Model) Model which we want save
    model_filepath: (string) Filepath where we want save model.

    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()