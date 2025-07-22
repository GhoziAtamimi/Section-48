import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import Custom_Exception
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())]
            )
            logging.info("Standard Scaler process for numerical columns is completed")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("oh",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))]
            )
            logging.info("Encoding Process for categorical columns is completed")

            preprocessor = ColumnTransformer(
                [("num_pipeline",num_pipeline,numerical_columns),
                 ("cat_pipeline",cat_pipeline,categorical_columns)]
            )
            
            return preprocessor
        except Exception as e:
            raise Custom_Exception(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

            logging.info("Process to read train and test data completed")

            preprocessing_obj = self.get_data_transformer()
            target_column = "math_score"
            numerical_columns = ["writing_score","reading_score"]
            
            train_input_features = df_train.drop(columns=[target_column],axis=1)
            train_target_feature = df_train[target_column]

            test_input_features = df_test.drop(columns=[target_column],axis=1)
            test_target_feature = df_test[target_column] 

            train_input_features_array = preprocessing_obj.fit_transform(train_input_features)
            test_input_features_array = preprocessing_obj.transform(test_input_features)

            train_array = np.c_[train_input_features_array,np.array(train_target_feature)]
            test_array = np.c_[test_input_features_array,np.array(test_target_feature)]

            logging.info(f"saved preprocessing object.")

            save_object(
                file_path = self.transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_array,
                test_array,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise Custom_Exception(e,sys)