import tensorflow as tf
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from source_code.neural_network_ml_classifier.data_processor.PreProcess import PreProcessor
from source_code.logistic_regression.inference import LRBasedFacultyClassifier
from source_code.neural_network_ml_classifier.classify.inference import NNBasedFacultyClassifier
from source_code.XGboost.inference import XGBFacultyClassifier
import os
import sys

data_file = crawled_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../Crawl-n-Extract/Merge/UIUC.txt')

"""
Entry point class for classification module

logit(default) --> Logistic Regression model
xgb --> XGBoost model
nn --> Custom Neural Network model

"""
class FacultyClassifier:

    def __init__(self, classifier_type='logit'):
        if classifier_type == 'logit':
            self.classifier = LRBasedFacultyClassifier()
        if classifier_type == 'nn':
            self.classifier = NNBasedFacultyClassifier() # This is not easy to run as there are additional steps to download and place the model files in correct location
        if classifier_type == 'xgb':
            self.classifier = XGBFacultyClassifier()

    def predict(self, data, print_pred=False):
        return self.classifier.predict(data, print_pred)


if __name__ == "__main__":
    classifier = sys.argv[1] if len(sys.argv) > 1 else 'logit'
    clf = FacultyClassifier(classifier)
    data = open(data_file, 'r').readlines()
    clf.predict(data, True)