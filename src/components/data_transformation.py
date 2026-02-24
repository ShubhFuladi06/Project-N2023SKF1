import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer # to perform column transformations such as scaling and encoding on the dataset
from sklearn.impute import SimpleImputer # to handle missing values in the dataset by replacing them with a specified strategy (e.g., mean, median, most frequent)
from sklearn.pipeline import Pipeline # to create a pipeline that sequentially applies a list of transformations and a final estimator
from sklearn.preprocessing import OneHotEncoder,StandardScaler # to convert categorical features into a format that can be provided

from src.exception import CustomException
from src.logger import logging        
import os     

from src.utils import save_object

@dataclass  # dataclass is a decorator that automatically generates special methods like __init__() and __repr__() for the class, making it easier to create classes that primarily store data.
class DataTransformationConfig: # to define the configuration for data transformation, including the path to save the preprocessor object
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl") # to specify the file path where the preprocessor object will be saved after fitting and transforming the data
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig() # to initialize the data transformation configuration

    def get_data_transformer_object(self):
        """This function is responsible for performing data transformation on the dataset.
        It defines the numerical and categorical columns, creates pipelines for both types of columns,
        and combines them into a preprocessor object that can be used to fit and transform the data."""
        
        try:
            numerical_columns=['writing_score','reading_score'] # to specify the numerical columns in the dataset that require scaling
            categorical_columns=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"] # to specify the categorical columns in the dataset that require encoding
        
            num_pipeline=Pipeline(
                steps=[     
                    ("imputer",SimpleImputer(strategy="median")), # to handle missing values in numerical columns by replacing them with the median value
                    ("scaler",StandardScaler()) # to scale the numerical features to have a mean of 0 and a standard deviation of 1
                ]
            )   
                
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")), # to handle missing values in categorical columns by replacing them with the most frequent value
                    ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore")) # to convert categorical features into a format that can be provided to machine learning algorithms
                ]
            )
        
            logging.info("Numerical columns scaling completed")
            logging.info("Categorical columns encoding completed")
            
            preprocessor=ColumnTransformer(
                transformers=[
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path) # to read the training dataset from the specified path
            test_df=pd.read_csv(test_path) # to read the testing dataset from the specified path
            
            logging.info("Read train and test data completed")
            
            logging.info("Obtaining preprocessor object")
            preprocessor_obj=self.get_data_transformer_object() # to obtain the preprocessor object that can be used to fit and transform the data
            
            target_column_name="math_score" # to specify the target column name in the dataset
            
            input_feature_train_df=train_df.drop(columns=[target_column_name]) # to separate the input features from the target variable in the training dataset
            target_feature_train_df=train_df[target_column_name] # to extract the target variable from the training dataset
            
            input_feature_test_df=test_df.drop(columns=[target_column_name]) # to separate the input features from the target variable in the testing dataset
            target_feature_test_df=test_df[target_column_name] # to extract the target variable from the testing dataset
            
            logging.info("Applying preprocessing object on training and testing datasets")
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df) # to fit and transform the input features of the training dataset using the preprocessor object
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df) # to transform the input features of the testing dataset using the preprocessor object
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)] # to concatenate the transformed input features and target variable of the training dataset into a single array
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)] # to concatenate the transformed input features and target variable of the testing dataset into a single array
            
            logging.info("Saved preprocessing object.")
             
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path, # to specify the file path where the preprocessor object will be saved
                obj=preprocessor_obj # to specify the preprocessor object that will be saved
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)