from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import pickle

from xgboost import XGBClassifier
import joblib
from PreProcessor import PreProcessor

import sys
"""
    This script reads the crawled data from path defined variable "crawled_data_path", trained XGBoost model.
    Uses the trained model to classify faculty pages.
    
    :input locations crawled_data_path: File path where crawled data is stored.
"""
crawled_data_path = '../Crawl-n-Extract/Merge/UIUC.txt'

# As we use same tf-idf vectorizer as used by our NN model, so loading the same here.
vectorizer:TfidfVectorizer = pickle.load(open('../data/improved_vectorized_data/vectorizer_object_new', 'rb'))

# Loading the saved model
xgb_model = joblib.load('xgb.model')

def run_inference(data_file=crawled_data_path):
    print(f"Loading data for inference from location: {data_file}")

    data = open(data_file, 'r').readlines()
    print(f"Loading crawled data completed. # docs read: {len(data)}")

    pp = PreProcessor()
    processed_lines = []
    counter = 0
    data_len = len(data)
    for line in data:
        line_split = line.split('#####')
        if(len(line_split) < 2):
            continue

        processed_line = pp.intersectStopWordsAndStem(line_split[1].strip())
        processed_lines.append(processed_line)
        counter += 1
        if counter % 100 == 0:
            print(f"Pre-processing data. Completed: {counter}/{data_len}")

    print(f"Pre-processing data. Completed: {data_len}/{data_len}")

    print("Working on classifying the faculty pages using pre-trained xgboost model..")
    
    processed_line_vec = vectorizer.    transform(processed_lines)
    predicted_values = xgb_model.predict(processed_line_vec)
    predicted_values_labeled = [round(value) for value in predicted_values]

    faculty_count = 0
    print("Printing the pages classified as faculty pages..")
    for idx in range(len(predicted_values_labeled)):
        if predicted_values_labeled[idx] == 1:
            faculty_count += 1
            print(data[idx].split('#####')[0])

if __name__ == "__main__":
    run_inference(sys.argv[1] if len(sys.argv) > 1 else crawled_data_path)