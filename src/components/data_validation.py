import os, sys 
import numpy as np 
import pandas as pd 
from typing import Optional
from scipy.stats import ks_2samp
from src.logger import logging
from  src.exception import CustomException
from src.utils import write_yaml_file
from src.entity import config_entity, artifact_entity


class DataValidation:

    def __init__(self, data_validation_config:config_entity.DataValidationConfig, data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        print(f"{'-'*20} Data Validation {'-'*20}")
        logging.info(f"{'='*20} Data Validation {'='*20}")
        self.data_validation_config = data_validation_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.validation_error = dict()

    def drop_missing_value_columns(self, df:pd.DataFrame, report_key:str):
        """
        Drop Columns with Missing values over the threshold parameter

        df: Dataframe Input Variable
        threshold: Percentage criteria to drop a column
        returns Pandas DataFrame
        ===============================================
        

        """
        try:
            threshold = self.data_validation_config.missing_values_threshold
            missing_value_report = df.isnull().sum()
            
            drop_columns_list = missing_value_report[missing_value_report > threshold].index
            logging.info(f"Filtering Columns with Null values above threshold: {threshold}, Columns: {drop_columns_list}")
            df.drop(drop_columns_list, axis=1, inplace=True)
            if len(df.columns) == 0:
                logging.info("Dataset does not meet the minimum criteria")
                return None 
            return df 
        except Exception as e:
            raise CustomException(e, sys)

    def is_required_columns_exists(self, base_df: pd.DataFrame, current_df:pd.DataFrame, report_key:str)-> bool:
        try:
            base_columns = base_df.columns 
            current_columns = current_df.columns 
            missing_columns = []

            for column in base_columns:
                if column not in current_columns:
                    missing_columns.append(column)
            
            if len(missing_columns) > 0:
                self.validation_error[report_key] = missing_columns
                return False
            return True

        except Exception as e:
            raise CustomException(e, sys)

    def data_drift(self, base_df: pd.DataFrame, current_df:pd.DataFrame, report_key:str):
        try:
            drift_rec = dict()
            base_columns = base_df.columns 
            current_columns = current_df.columns
            for column in base_columns:
                base_data, current_data = base_df[column], current_df[column]
                column_distribution = ks_2samp(base_data, current_data)

                if column_distribution.pvalue > 0.05:
                    drift_rec[column] = {
                        'p_value': column_distribution.pvalue,
                        'same_distribution': True
                    }
                else:
                    drift_rec[column] = {
                        'p_value': column_distribution.pvalue,
                        'same_distribution': False
                    }
                self.validation_error[report_key] = drift_rec

        except Exception as e:
            raise CustomException(e, sys)

    
    def initiate_data_validation(self)-> artifact_entity.DataValidationArtifact:
        try:
            base_df = pd.read_csv(self.data_validation_config.base_file_dir)
            cat_cols = [col for col in base_df.columns if base_df[col].dtype == 'O']
            for col in cat_cols:
                base_df[col] = base_df[col].apply(lambda x: x.strip())
            base_df.replace(to_replace=['na', '?'], value=np.NAN, inplace=True)
            logging.info("Validating the Missing Values Against the Base Dataset")
            base_df = self.drop_missing_value_columns(df=base_df, report_key='missing_values_base')
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_dir)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_dir)

            logging.info("Validating the Missing Values Against the Train Dataset")
            train_df = self.drop_missing_value_columns(df=train_df, report_key='missing_values_train')
            logging.info("Validating the Missing Values Against the Test Dataset")
            test_df = self.drop_missing_value_columns(df=test_df, report_key='missing_values_test')

            logging.info("Validating the Features of the Dataset")
            train_status = self.is_required_columns_exists(base_df=base_df, current_df=train_df, report_key='missing+columns_train')
            test_status = self.is_required_columns_exists(base_df=base_df, current_df=test_df, report_key='missing+columns_test')

            if train_status:
                self.data_drift(base_df=base_df, current_df=train_df, report_key='data_drift_train')
            if test_status:
                self.data_drift(base_df=base_df, current_df=test_df, report_key='data_drift_test')

            logging.info("Composing the Validation Report..")
            write_yaml_file(file_path=self.data_validation_config.report_file_dir, data=self.validation_error)
            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_dir=self.data_validation_config.report_file_dir)

            logging.info(f"{'='*20} Exiting Data Validation {'='*20}")
            return data_validation_artifact    


        except Exception as e: 
            raise CustomException(e, sys)

