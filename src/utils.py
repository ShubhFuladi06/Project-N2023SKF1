import os
import sys
import numpy as np
import pandas as pd     
from src.exception import CustomException
from src.logger import logging
import dill          # dill is a Python library that provides a way to serialize and deserialize Python objects, allowing you to save and load complex data structures, including custom classes and functions, to and from files. It is an extension of the built-in pickle module with additional features and support for a wider range of Python objects.

from sklearn.metrics import r2_score # to evaluate the performance of regression models by calculating the R2 score, which measures the proportion of variance in the target variable that can be explained by the features in the model


def save_object(file_path: str, obj: object):
    try:
        dir_path = os.path.dirname(file_path) # to get the directory path from the file path
        os.makedirs(dir_path, exist_ok=True) # to create the directory if it doesn't exist

        with open(file_path, "wb") as file_obj: # to open the file in binary write mode
            dill.dump(obj, file_obj) # to serialize the object and save it to the file using dill

    except Exception as e:
        raise CustomException(e, sys) # to raise a custom exception if any error occurs during the saving process


def evaluate_models(X_train, y_train, X_test, y_test, models,param):
    try:
        report = {} # to initialize an empty dictionary to store the evaluation report of the models

        for i in range(len(models)):
            model = list(models.values())[i] # to get the model object from the models dictionary
            model.fit(X_train, y_train) # to fit the model on the training data

            y_train_pred = model.predict(X_train) # to make predictions on the training data using the fitted model
            y_test_pred = model.predict(X_test) # to make predictions on the testing data using the fitted model

            train_model_score = r2_score(y_train, y_train_pred) # to calculate the R2 score for the training data
            test_model_score = r2_score(y_test, y_test_pred) # to calculate the R2 score for the testing data

            report[list(models.keys())[i]] = test_model_score # to store the R2 score of the testing data in the report dictionary with the model name as the key

        return report # to return the evaluation report of all models
    
    except Exception as e:
        raise CustomException(e, sys) # to raise a custom exception if any error occurs during the evaluation process