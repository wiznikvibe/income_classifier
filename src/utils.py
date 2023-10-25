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