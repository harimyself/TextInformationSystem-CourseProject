Trained neural network model is 187MB. Git has upper cap of 100MB.
Pleae downlaod the trained neural network model from the link below

Step 1: Download the folder 'neural_network_model_v2' from the link below.
https://uofi.box.com/v/es-neural-network-model-v2

Step 2: Ensure that directories 'model', 'vectorizer' are placed directly under 'fully_trained_model'.

Step 3: Enusre below are directories are safely exctracted.
    'fully_trained_model/model'
    'fully_trained_model/vectorizer'

Step 4: Point 'crawled_data_path' in the script 'classify/infer_crawled_data.py' point to your webpage dump.
        Note: The carwled data should have one doc per line. Separated by  '#####'

Step 5: Run the script 'infer_crawled_data' and observe the faculy links printed in console.