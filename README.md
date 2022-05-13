# Course-094295_hw1
Early Prediction of Sepsis from Clinical Data - First homework assignment in the "Data Presentation and Analysis" Technion course.

LSTM folder: <br>
Contains the trainer class file, patient dataset class and the
training and deployment file for the LSTM model. <br>

LogisticRegression folder: <br>
Contains the training and deployment file for the LR model. <br>

notebooks folder: <br>
Contains notebooks we used for exploratory data analysis, processing and predictions. <br>
Mainly used for personal checks, and is not part of any running process. <br>

utils folder: <br>
Contains score functions and training utilities. <br>

models folder: <br>
Contains trained models and scalers. <br>

preprocess_data.py: <br>
Preprocessing the data and saving it to a new csv file. <br>

predict_LR.py: <br>
Using LR model for prediction, saving predictions as csv. <br>

predict_LSTM.py: <br>
Using LSTM model for prediction, saving predictions as csv. <br>

predict.py: <br>
Run this file to use the prediction model (test_directory is the test data path). <br>
This uses the LSTM model. If you want to use the LR model, add parameter model_type and set as 'LR'. <br>
