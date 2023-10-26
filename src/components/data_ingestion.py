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
        logging.info(f"{'-'*20} Data Ingestion Process Complete {'-'*20}")
        

    def initiate_data_ingestion(self)-> artifact_entity.DataIngestionArtifact:
        try:
            logging.info("Exporting Server data into a DataFrame")
            data = get_collection_dataframe(database_name=self.data_ingestion_config.database_name, collection_name=self.data_ingestion_config.collection_name)

            logging.info(f"Features of the Dataset Include: {data.columns}")
            
            cat_cols = [col for col in data.columns if data[col].dtype == 'O']
            for col in cat_cols:
                data[col] = data[col].apply(lambda x: x.strip())

            data.replace(to_replace=['na', '?'], value=np.NAN, inplace=True)
            logging.info("Storing the raw files in the Feature Directory")
            
            feature_store_path = os.path.dirname(self.data_ingestion_config.feature_store_dir)
            os.makedirs(feature_store_path, exist_ok=True)
            data.to_csv(path_or_buf=self.data_ingestion_config.feature_store_dir, index=False, header=True)

            logging.info("Splitting the Raw data for Model Building and Data Transformation")
            train_df, test_df = train_test_split(data, test_size=self.data_ingestion_config.test_size, random_state=42)

            train_data_path = os.path.dirname(self.data_ingestion_config.train_data_dir)
            os.makedirs(train_data_path, exist_ok=True)

            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_data_dir, index=False, header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_data_dir, index=False, header=True)

            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_dir=self.data_ingestion_config.feature_store_dir,
                train_file_dir=self.data_ingestion_config.train_data_dir,
                test_file_dir=self.data_ingestion_config.test_data_dir
            )

            logging.info(f"{'-'*20} Data Ingestion Process Complete {'-'*20}")

            return data_ingestion_artifact


        except Exception as e:
            raise CustomException(e, sys)