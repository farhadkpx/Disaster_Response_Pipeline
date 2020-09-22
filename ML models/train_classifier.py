import sys
import numpy as np
import pandas as pd
import pickle
import re
import nltk
import argparse

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import shuffle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
#---------------------------------------------------------------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score,\
 recall_score, precision_score, f1_score
#---------------------------------------------------------------------

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier

nltk.download(['punkt', 'wordnet'])

def file_processing():
    """
    There are two default arguments: database_filepath
    and model_filepath. database_filepath contains the path to the
    SQLite database containing the messages. model_filepath contains the
    path to the trained model. An optional argument --model_type controls
    which model to use. The default model type is an RandomForest classifier model using 
    Tfidf Vectorizer to transform the messages.
    
    RETURNS:
        args - The arguments inserted by the user
    """
    parser = argparse.ArgumentParser()

    # Default arguments
    parser.add_argument("database_filepath", type=str,\
                        help="Path of database")
    parser.add_argument("model_filepath", type=str,\
                        help="Path of trained model")
    # Optional arguments
    parser.add_argument("--model_type", default=1, type=int, choices={1},\
                        help="Choose 1 for RandomForest Classifier with Tfidf Vectorizer")
    
    args = parser.parse_args()

    return args


def load_data(database_filepath):
    """
    Loads the SQLite database from the given database_filepath. Divide the
    data into model inputs and message labels.
    
    INPUTS:
        database_filepath - path to the SQLite database containing the messages
    RETURNS:
        X - inputs to be used for modeling. Contains the messages.
        Y - labels for modeling. Contains the categories of the messages
        category_names - list conatining all types of message categories
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    
    # Generate data and labels
    df = pd.read_sql_table('messages', engine)
    
    # selecting target and feature variables > 'related' to 'direct_report'
    X = df['message']
    Y = df.loc[:, 'related':'direct_report']
    
    # Get names of all categories
    category_names = Y.columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
    """
    Clean and tokenize text for modeling. It will replace all non-
    numbers and non-alphabets with a blank space. Next, it will
    split the sentence into word tokens and lemmatized them with Nltk's 
    WordNetLemmatizer(), first using noun as part of speech, then verb.
    Finally, the word tokens will be stemmed with Nltk's PorterStemmer.
    
    INPUTS:
        text - the message to be clean and tokenized
    RETURNS:
        clean_tokens: the list containing the cleaned and tokenized words of
                        the message
    """
     # replace all non-alphabets and non-numbers with blank space
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # instantiate stemmer
    stemmer = PorterStemmer()
    
    clean_tokens = []
    for tok in tokens:
        # lemmtize token using noun as part of speech
        clean_tok = lemmatizer.lemmatize(tok)
        # lemmtize token using verb as part of speech
        clean_tok = lemmatizer.lemmatize(clean_tok, pos='v')
        # stem token
        clean_tok = stemmer.stem(clean_tok)
        # strip whitespace and append clean token to array
        clean_tokens.append(clean_tok.strip())
        
    return clean_tokens


def build_model(model_type):
    """
    Builds the pipeline that will transform the messages and the model them
    based on the user's model selection. It will also perform a grid search
    to find the optimal model parameters.
    
    INPUTS:
        model_type - the model type selected by the user.
    RETURNS:
        cv  - the model with the best parameters as determined by grid
                    search
    """
    if model_type == 1:
        pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(tokenizer=tokenize)),\
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
                ])
                
        # setting up hyper-parameter tuning...grid search
        parameters = {
        'tfidf__max_df': (0.8, 1.0),      
        'tfidf__ngram_range': ((1, 1),(1,3)), 
        'clf__estimator__n_estimators': (25, 50)      
        }     
    model_cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=2, n_jobs=-1)
    return model_cv

#-----------------------------------------------------------
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model's performance on the test data. Will return the
    precision, recall and f1 score for each category.
    
    INPUTS:
        model - the optimized model used to classify messages
        X_test - the model inputs of the test data
        Y_test - the labels of the messages from the test data
        category_names - the names of all message categories
    """
    Y_pred = model.predict(X_test)

    # Get names of all categories
    category_names = Y_test.columns.tolist()

    Y_pred_df = pd.DataFrame(Y_pred, columns = category_names)
    
    for i, col in enumerate(category_names):        
     #if i in [0, 9]: continue   Skip bad column, TODO: Fix 0th column to not be 3 classes for no reason
        print("Column {}: {}".format(i, col))
    
        y_true = list(Y_test.iloc[:, i])
        y_pred = list(Y_pred_df.iloc[:, i])
        print('Accuracy of message: %25s: %.2f' %(category_names[i], accuracy_score(y_true, y_pred)))
        
        target_names = ['is_{}'.format(col), 'is_not_{}'.format(col)] 
        print(classification_report(y_true, y_pred, target_names=target_names))
        
    
            
def save_model(model, model_filepath):
    """
    Save the optimized model to the path specified by model_filepath
    
    INPUTS:
        model - the optimized model
        model_filepath - the path where the model will be saved
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    print(len(sys.argv))
    if ((len(sys.argv) == 3) or (len(sys.argv) == 5)):
        args = file_processing()
        database_filepath = args.database_filepath
        model_filepath = args.model_filepath
        model_type = args.model_type
       
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
       
        X_train, X_test, Y_train, Y_test =\
        train_test_split(X, Y, test_size=0.2, shuffle=True)
        
        print('Building model...')
        model = build_model(model_type)
        
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