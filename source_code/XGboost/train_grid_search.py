from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, GridSearchCV

import os
import pickle


"""
    This is the script we have used to do hyperparameter tuning for XGBoost.

    Note: This script takes a long time to run. 
"""

PROCESSED_DATA_BASE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/../data/classificationData/'

def load_data():
        print("Starting to loading train data..")
        #X is already transfored with TfidfVectorizer
        X_train = pickle.load(open(PROCESSED_DATA_BASE_PATH + 'train/X', 'rb'))
        X_train = X_train.todense()
        Y_train = pickle.load(open(PROCESSED_DATA_BASE_PATH + 'train/Y', 'rb'))
        print("Finished loading train data..")

        print("Starting to loading test data..")
        #X is already transfored with TfidfVectorizer
        X_test = pickle.load(open(PROCESSED_DATA_BASE_PATH + 'test/X', 'rb'))
        X_test = X_test.todense()
        Y_test = pickle.load(open(PROCESSED_DATA_BASE_PATH + 'test/Y', 'rb'))
        print("Finished loading test data..")

        return X_train, Y_train, X_test, Y_test

def train_grid_search():
    X_train, Y_train, X_test, Y_test = load_data()

    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}, X_test: {X_test.shape}, Y_test: {Y_test.shape}")

    xgb_params = {
        'nthread' : [4],
        'random_state' :[23],
        'seed': [2],
        'colsample_bytree' : [0.6, 0.7],
        'subsample' : [0.7, 0.8],
        'tree_method' : ['hist'],
        'max_depth' : [5, 10], 
        'n_estimators' : [5, 10],
        'objective' : ['binary:logistic']
    }

    label_encoder = LabelEncoder().fit(Y_train)
    label_encoder_y_train = label_encoder.transform(Y_train)
    label_encoder_y_test =  label_encoder.transform(Y_test)
    # fit model no training data
    xgb_model = XGBClassifier()


    clf = GridSearchCV(xgb_model, xgb_params, n_jobs=5, 
                   scoring='roc_auc',
                   verbose=2, refit=True)

    clf.fit(X_train, label_encoder_y_train.ravel())



    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    print('Raw AUC score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

if __name__ == "__main__":
    train_grid_search()
