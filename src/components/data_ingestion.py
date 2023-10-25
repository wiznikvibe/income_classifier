import os, sys 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from src.entity import config_entity, artifact_entity
from src.utils import get_collection_dataframe

class DataIngestion:

    def __init__(self, data_ingestion_config:config_entity.DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        print(f"{'='*20} Data Ingestion {'='*20}")

    def initiate_data_ingestion(self)-> artifact_entity.DataIngestionArtifact:
        try:
            logging.info("Exporting Server data into a DataFrame")
            data = get_collection_dataframe(database_name=self.data_ingestion_config.database_name, collection_name=self.data_ingestion_config.collection_name)

            logging.info(f"Features of the Dataset Include: {data.columns}")
            
            cat_cols = [col for col in data.columns if data[col].dtype == 'O']
            for col in cat_cols:
                data[col] = data[col].apply(lambda x: x.strip())

            data.replace(to_replace=['na', '?'], value=np.NAN, inplace=True)
             


        except Exception as e:
            raise CustomException(e, sys)