import yaml
import dill
import os, sys 
import pandas as pd 
import numpy as np 
from src.logger import logging
from src.exception import CustomException
from src.config import mongo_client


def get_collection_dataframe(database_name:str, collection_name:str)->pd.DataFrame:
    """
    This Function retrieves the dataset from the backend server
    """
    try:
        logging.info(f"Reading Data From DataBase: {database_name}, Collection: {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"Data Rows: {df.shape[0]} || Columns: {df.shape[1]}")
        if "_id" in df:
            df = df.drop("_id", axis=1)
        
        return df

    except Exception as e:
        raise CustomException(e, sys)


def write_yaml_file(file_path:str, data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)
        with open(file_path, "w") as file_obj:
            yaml.dump(data, file_obj)
            file_obj.close()

    except Exception as e:
        raise CustomException(e, sys)


def save_numpy_data(file_dir: str, array: np.array):
    try:
        logging.info("Data is Passed to Utils function for Saving the processed data")
        os.makedirs(os.path.dirname(file_dir), exist_ok=True)
        with open(file_dir, "wb") as file_obj:
            np.save(file_obj, array)
            file_obj.close()
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_dir: str, obj: object)-> None:
    try:
        logging.info("Data is Passed to Utils function for Saving the processed data")
        os.makedirs(os.path.dirname(file_dir), exist_ok=True)
        with open(file_dir, "wb") as file_obj:
            dill.dump(obj, file_dir)
            file_dir.close()
            
    except Exception as e:
        raise CustomException(e, sys)