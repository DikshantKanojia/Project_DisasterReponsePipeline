# import necessary libraries
import sys
import pickle
import re
import pandas as pd
import numpy as np
import nltk
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD

def load_data(database_filepath):
    """
    Function to load Data from a database

    Args:
        database_filepath: file path of the database

    Output:
        Load the data file and return it as predictor and response variable.
        Also, return the catogory names
    """

    # Load data from database
    table_name = 'messages_disaster'
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table(table_name,engine)

    # Predictor Variable
    X = df['message']

    # Response Variable
    y = df.drop(['id', 'message', 'genre', 'original'], axis = 1)

    # Category name
    category_names = y.columns

    return X, y, category_names

def tokenize(text):
    """
    Tokenizes text data

    Args:
        text: Messages as input

    Output:
        word list: list of words which are normalized, and in their root form.
    """

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Substitute everything that is not alphabets or numbers with space

    # tokenize text
    words = word_tokenize(text)

    # remove stop words
    stopWords = stopwords.words("english") # What does this do
    words = [word for word in words if word not in stopWords]

    # extract root form of words
    words = [WordNetLemmatizer().lemmatize(word, pos = 'v') for word in words]

    return words


def build_model():
    """
    Builds model, create pipeline, and hypertunes parameters using gridsearch

    Args: N/A

    Output: Returns the model
    """

    # Create a pipeline
    pipeline = Pipeline([
                      ('vect', CountVectorizer(tokenizer = tokenize)),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultiOutputClassifier(RandomForestClassifier()))

                    ])

    # Parameters for hypertuning
    parameters = {'tfidf__use_idf': (True, False),
              'clf__estimator__n_estimators': [50, 100],
              'clf__estimator__min_samples_split': [2, 4]}

    # Perform GridSearch
    cv = GridSearchCV(pipeline, param_grid = parameters)

    return cv



def evaluate_model(model, X_test, y_test, category_names):
    """
    Function to evaluate the model

    Args:
        model: the machine learning model to be used
        X_test: Predictor test dataset
        Y_test: Response test dataset
        category_names: Different classification categories

    Output:
        Displays the classification report and accurary score
    """

    # Make predictions
    y_pred = model.predict(X_test)

    # Print classifcation report
    for i, column_name in enumerate(y_test):
        print(column_name)
        print(classification_report(y_test[column_name], y_pred[:, i])) # print all rows and ith column

    # Print Accuracy Score
    print('Accuracy Score: {}'.format(np.mean(y_test.values == y_pred)))



def save_model(model, model_filepath):
    """
    Function to save the model

    Args:
        model: the model to be saved
        model_filepath: Filepath where the model will be saved

    Output:
        Saves the model as a pickle file
    """

    # Save the model
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
        evaluate_model(model, X_test, Y_test, category_names)

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
