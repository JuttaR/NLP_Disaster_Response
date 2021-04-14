import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import pickle
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
import sys

nltk.download(['punkt', 'stopwords', 'wordnet'])


def load_data(database_filepath):
    """
    Reads in data from SQLite database and outputs dataframes for ML model

    INPUT:
        database_filepath: filepath to SQLite database

    OUTPUTS:
        X: features dataframe
        y: target dataframe
        category_names: names of category targets
    """
    # create SQLAlchemy engine and read in data
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('df', engine)

    # create features (X) and target (y) dataframes
    X = df['message']
    y = df.iloc[:, 4:]

    # retrieve names of targets
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    """
    Cleans text data in order to use it for machine learning later.

    INPUT:
        text: Raw message

    OUTPUT:
        cleaned_tokens: Cleaned text (w/o urls, normalized, tokenized, w/o stopwords, lemmatized) as list
    """
    # Use regex expression to detect urls in text
    url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    detected_urls = re.findall(url_regex, text)

    # Replace all urls with placeholder
    for detected_url in detected_urls:
        text = text.replace(detected_url, "url")

    # Normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenization
    words = word_tokenize(text)

    # Removal of stopwords
    words = [w for w in words if w not in stopwords.words("english")]

    # Lemmatization (nouns)
    cleaned_tokens = [WordNetLemmatizer().lemmatize(w).strip() for w in words]

    return cleaned_tokens


def build_model():
    """
    Builds a ML Pipeline using a RandomForest classifier and GridSearchCV to tune it to its optimal hyperparameters.

    INPUT:
        None

    OUTPUT:
        model: GridSearch output from cross-validation
    """
    # set up of ML Pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])

    # set up of hyper-parameter dictionary for GridSearchCV of RandomForestClassifier
    params = {
        'clf__estimator__min_samples_split': [2, 4],  # min number of data points in node before the node is split
        'clf__estimator__max_features': ['log2', 'auto'],  # max number of features considered for splitting a node
        'clf__estimator__n_estimators': [50, 100],  # number of trees in the forest
    }

    # create model
    model = GridSearchCV(pipeline, param_grid=params, scoring='accuracy', verbose=3, n_jobs=1, cv=3)
    return model


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate trained ML model performance against test data.

    INPUTS:
        model: Trained ML model
        X_test: Test data (features)
        y_test: Test data (targets / labels)
        target_names: Names of labels

    OUTPUT:
        Printed results from GridSearch; classification report and accuracy for all categories
    """
    # print results from GridSearch
    model_results_table = pd.concat([pd.DataFrame(model.cv_results_["params"]),
                                     pd.DataFrame(model.cv_results_["mean_test_score"],
                                                  columns=["Accuracy"])], axis=1)

    print("GridSearch Results Table")
    print(model_results_table)

    print("Best-performing parameters from GridSearch:", model.best_params_)

    # make predictions on test data
    y_pred = model.predict(X_test)

    # print classification reports
    print("Classification report incl. overall micro, macro, weighted and sample averages")
    print(classification_report(y_test, y_pred, target_names=category_names, zero_division=0))

    print("Individual classification report incl. accuracy, macro and weighted averages")
    y_pred_df = pd.DataFrame(y_pred, columns=y_test.columns)
    for category in y_test.columns:
        print('-------------------------------------------------\n Category: {}\n'.format(category))
        print(classification_report(y_test[category], y_pred_df[category], zero_division=0))


def save_model(model, model_filepath):
    """
    Save trained model to pickle file

    INPUTS:
        model: trained model
        model_filepath: filepath to save model as pickle file (byte stream)

    OUTPUT:
        pickle file // none
    """
    # save trained model in binary
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/Disaster.db classifier.pkl')


if __name__ == '__main__':
    main()