import os, sys
import numpy as np 
import pandas  as pd 
from src.exception import CustomException
from src.logger import logging 
from src.entity import config_entity, artifact_entity
from src import utils 
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class DataTransformation:

    def __init__(self, data_transformation_config:config_entity.DataTransformationConfig, data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        print(f"{'='*20} Data Transformation {'='*20}")
        logging.info(f"{'='*20} Data Transformation {'='*20}")
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    
    @classmethod 
    def get_data_transformer_obj(cls, num_cols, cat_cols)->ColumnTransformer:
        try:
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean',fill_value=0)),
                ('scaler', RobustScaler())
            ])
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder',OneHotEncoder(handle_unknown='ignore', sparse_output=False)),   
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num" , numerical_pipeline, num_cols),
                ("cat", categorical_pipeline, cat_cols)
            ])

            return preprocessor            
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self,)-> artifact_entity.DataTransformationArtifacts:
        try:
            logging.info(f"{'='*20} Data Transformation {'='*20}")
            logging.info("Reading the Training and Testing Data")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_dir)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_dir)

            logging.info("Segregating the Input Features and Output Features")
            input_train_df = train_df.drop(self.data_transformation_config.target_column, axis=1)
            input_test_df = test_df.drop(self.data_transformation_config.target_column, axis=1)
            
            train_target = train_df[self.data_transformation_config.target_column]
            test_target = test_df[self.data_transformation_config.target_column] 

            logging.info("Applying Encoder on the Target Feature")
            label = LabelEncoder()
            label.fit(train_target)
            train_target_arr = label.transform(train_target)
            test_target_arr = label.transform(test_target)


            logging.info("Passing the Input Data into Transformer")
            numerical_features = [col for col in input_train_df.columns if input_train_df[col].dtype == 'int']
            categorical_features = [col for col in input_train_df.columns if input_train_df[col] not in numerical_features]

            transformation_pipeline = DataTransformation.get_data_transformer_obj(num_cols=numerical_features,cat_cols=categorical_features)
            transformation_pipeline.fit(input_train_df)
            input_train_arr = transformation_pipeline.transform(input_train_df)
            input_test_arr = transformation_pipeline.transform(input_test_df)
            
            logging.info("Resampling the Dataset to generate samples for Minority class")
            smt = SMOTETomek(sampling_strategy='minority', random_state=42)
            
            logging.info(f"Before Resampling - Train Data: {input_train_arr.shape}, Target Data: {train_target_arr.shape}")
            train_input_arr, target_train_arr = smt.fit_resample(input_train_arr, train_target_arr)
            logging.info(f"After Resampling - Train Data: {train_input_arr.shape} || Target Data: {target_train_arr.shape}")

            logging.info(f"Before Resampling - Train Data: {input_test_arr.shape}, Target Data: {test_target_arr.shape}")
            test_input_arr, target_test_arr = smt.fit_resample(input_test_arr, test_target_arr)
            logging.info(f"After Resampling - Train Data: {test_input_arr.shape} || Test Data: {target_test_arr.shape}")

            train_arr = np.c_[train_input_arr, target_train_arr]
            test_arr = np.c_[test_input_arr, target_test_arr]

            logging.info("Saving the Processed Data")
            utils.save_numpy_data(file_dir=self.data_transformation_config.transform_train_dir, array=train_arr)
            utils.save_numpy_data(file_dir=self.data_transformation_config.transform_test_dir, array=test_arr)

            logging.info("Saving the Preprocessor Object and Target Encoder")
            utils.save_object(file_dir=self.data_transformation_config.transform_obj_dir, obj=transformation_pipeline)
            utils.save_object(file_dir=self.data_transformation_config.transform_obj_dir, obj=label)

            data_transformation_artifact = artifact_entity.DataTransformationArtifacts(
                transform_obj_dir=self.data_transformation_config.transform_obj_dir,
                transform_train_dir=self.data_transformation_config.transform_train_dir,
                transform_test_dir=self.data_transformation_config.transform_test_dir,
                target_encoder_dir=self.data_transformation_config.target_encoder_dir
            )
            logging.info(f"{'='*20} Exiting Data Transformation {'='*20}")
            return data_transformation_artifact


        except Exception as e: 
            raise CustomException(e, sys)