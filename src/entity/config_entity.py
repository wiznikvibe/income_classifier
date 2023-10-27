import os, sys 
from src.exception import CustomException
from src.logger import logging 
from datetime import datetime 

FILE_NAME = "raw_data.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJ_FILE_NAME = "transformer.pkl"
TARGET_ENCODER_OBJ_FILE_NAME = "target_encoder.pkl"
MODEL_FILE_NAME = "model.pkl"

class TrainingPipelineConfig:

    def __init__(self):
        self.artifact_dir = os.path.join(os.getcwd(), "artifact", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")

class DataIngestionConfig:
    
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.database_name = "hr"
        self.collection_name = "details"
        self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, 'data_ingestion')
        self.feature_store_dir = os.path.join(self.data_ingestion_dir, "feature_store", FILE_NAME)
        self.train_data_dir = os.path.join(self.data_ingestion_dir, "dataset", TRAIN_FILE_NAME)
        self.test_data_dir = os.path.join(self.data_ingestion_dir, "dataset", TEST_FILE_NAME)
        self.test_size = 0.2

class DataValidationConfig:

    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir,"data_validation")
        self.report_file_dir = os.path.join(self.data_validation_dir, "report.yaml") 
        self.missing_values_threshold: fload = 0.2
        self.base_file_dir = os.path.join("salary.csv")


class DataTransformationConfig:

    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir,"data_transformation")
        self.target_column = 'salary'
        self.transform_obj_dir = os.path.join(self.data_transformation_dir,"transformer", TRANSFORMER_OBJ_FILE_NAME)
        self.transform_train_dir = os.path.join(self.data_transformation_dir,"transformer",TRAIN_FILE_NAME.replace('csv','npz'))
        self.transform_test_dir = os.path.join(self.data_transformation_dir,"transformer",TEST_FILE_NAME.replace('csv','npz'))
        self.target_encoder_dir = os.path.join(self.data_transformation_dir,"transformer", TARGET_ENCODER_OBJ_FILE_NAME)
         