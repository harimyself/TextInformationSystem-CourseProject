import tensorflow as tf
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
import numpy as np
from source_code.neural_network_ml_classifier.data_processor.PreProcess import PreProcessor

model_base_path = '../fully_trained_model/neural_network_model_v2/'

"""
    CLass to run Neural Network Based Faculty page classifier
"""
class NNBasedFacultyClassifier(object):

    def __init__(self, model_base_path=model_base_path):
        print("Loading trained model from: ", model_base_path + '/model')
        self.model:Sequential = tf.keras.models.load_model(model_base_path + '/model')

        print("Loading vectorized from: ", model_base_path + '/vectorizer/vectorizer_object')
        self.vectorizer:TfidfVectorizer = pickle.load(open(model_base_path + '/vectorizer/vectorizer_object', 'rb'))

        print("Loading trained models completed..")

    def predict(self, crawled_data, print_pred=False):
        print(f"Running inference using neural network model for {len(crawled_data)} pages")
        pp = PreProcessor()
        processed_lines = []
        counter = 0
        crawled_data_len = len(crawled_data)
        for line in crawled_data:
            line_split = line.split(' ##### ')
            if(len(line_split) < 2):
                continue

            processed_line = pp.intersectStopWordsAndStem(line_split[1])
            processed_lines.append(processed_line)
            counter += 1
            if counter % 100 == 0:
                print("Pre-processing data. Completed: ", counter, "/", crawled_data_len)

        print("Pre-processing data. Completed: ", crawled_data_len, "/", crawled_data_len)

        print("Working on classifying the faculty pages using pre-trained neural network model..")
        processed_line_vec = self.vectorizer.transform(processed_lines)
        predicted_values = self.model.predict(processed_line_vec)
        predicted_values_labeled = np.where(predicted_values > 0.5, 1, 0)

        if print_pred:
            faculty_count = 0
            print("Printing the pages classified as faculty pages..")
            for idx in range(len(predicted_values_labeled)):
                if predicted_values_labeled[idx][0] == 1:
                    faculty_count += 1
                    print(crawled_data[idx].split(" ##### ")[0])
                    return predicted_values