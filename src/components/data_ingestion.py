from src.exception import Custom_Exception
from src.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer
 

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path:str = os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Ingestion Process Start")
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Read dataset")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Split Data Process Start")

            train_data, test_data = train_test_split(df,test_size=0.2,random_state=42)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion Process End")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise Custom_Exception(e,sys)
        
if __name__ == "__main__":
    obj=DataIngestion()
    input_train_path,input_test_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,obj_path = data_transformation.initiate_data_transformation(train_path=input_train_path,test_path=input_test_path)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))