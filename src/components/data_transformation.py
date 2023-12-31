import os, sys
import numpy as np 
import pandas  as pd 
from src.exception import CustomException
from src.logger import logging 
from src.entity import config_entity, artifact_entity
from src import utils 
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class DataTransformation:

    def __init__(self, data_transformation_config:config_entity.DataTransformationConfig, data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        print(f"{'='*20} Data Transformation {'='*20}")
        logging.info(f"{'='*20} Data Transformation {'='*20}")
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    
    @classmethod 
    def get_data_transformer_obj(cls, num_cols, ordinal_cat_cols, nom_cat_cols, workclass_cat, education_cat)->ColumnTransformer:
        try:
            numerical_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='mean')),
                ('scale', RobustScaler())
            ])

            ordinal_cat_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,categories=[workclass_cat, education_cat])),
                ('scale', RobustScaler())
            ])

            nominal_cat_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('encode', OneHotEncoder(handle_unknown='ignore', sparse=False)),
                ('scaler', RobustScaler())
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('numerical', numerical_pipeline, num_cols),
                ('ordinal_enc', ordinal_cat_pipeline, ordinal_cat_cols),
                ('nominal_enc', nominal_cat_pipeline, nom_cat_cols)
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
            logging.info(f"Training Data Columns: {train_df.columns} || Testing Data Size: {test_df.shape}")    


            logging.info("Segregating the Input Features and Output Features")
            input_train_df = train_df.drop(self.data_transformation_config.target_column, axis=1)
            # input_train_df.drop('fnlwgt', axis=1, inplace=True)
            # input_train_df.drop(['capital-loss','capital-gain','fnlwgt'], axis=1, inplace=True)
            logging.info(input_train_df.shape)
            logging.info(f"Shape After Dropping the Target Value")
            input_test_df = test_df.drop(self.data_transformation_config.target_column, axis=1)
            # input_test_df.drop('fnlwgt', axis=1, inplace=True)
            # input_test_df.drop(['capital-loss','capital-gain','fnlwgt'], axis=1, inplace=True)

            train_target = train_df[self.data_transformation_config.target_column]
            logging.info(train_target[:5])
            test_target = test_df[self.data_transformation_config.target_column] 

            logging.info("Applying Encoder on the Target Feature")
            label = LabelEncoder()
            label.fit(train_target)
            train_target_arr = label.transform(train_target)
            test_target_arr = label.transform(test_target)
            logging.info(train_target[0])


            logging.info("Passing the Input Data into Transformer")

            ordinal_cat_cols = ['workclass', 'education']
            numerical_features = input_train_df.select_dtypes(exclude='object').columns
            nom_cat_features = [col for col in input_train_df.columns if col not in numerical_features and col not in ordinal_cat_cols]
            logging.info(f"Numerical Features: {numerical_features} | Nominal Features {nom_cat_features} | Ordinal Features: {ordinal_cat_cols}")
            logging.info(train_df.head)
            
            
            workclass_cat = [
                'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov',
                                'Self-emp-inc', 'Without-pay', 'Never-worked'
            ]

            education_cat = [
                'Doctorate', 'Masters', 'Bachelors', 'HS-grad',  'Some-college', 'Assoc-acdm',
                'Assoc-voc',  'Prof-school', '12th', '10th', '11th', '9th', '7th-8th',  '5th-6th',
                '1st-4th', 'Preschool'
            ]

            transformation_pipeline = DataTransformation.get_data_transformer_obj(num_cols=numerical_features, ordinal_cat_cols=ordinal_cat_cols, nom_cat_cols=nom_cat_features, workclass_cat=workclass_cat, education_cat=education_cat)
            transformation_pipeline.fit(input_train_df)
            input_train_arr = transformation_pipeline.transform(input_train_df)
            input_test_arr = transformation_pipeline.transform(input_test_df)
            logging.info(f"Shape of Data before Sampling: {input_train_arr.shape}")
            
            logging.info("Resampling the Dataset to generate samples for Minority class")
            smt = SMOTETomek(sampling_strategy='minority', random_state=10, n_jobs=-1)
            
            logging.info(f"Before Resampling - Train Data: {input_train_arr.shape}, Target Data: {train_target_arr.shape}")
            train_input_arr, target_train_arr = smt.fit_resample(input_train_arr, train_target_arr)
            logging.info(f"After Resampling - Train Data: {train_input_arr.shape} || Target Data: {target_train_arr.shape}")

            logging.info(f"Before Resampling - Test Data: {input_test_arr.shape}, Target Data: {test_target_arr.shape}")
            test_input_arr, target_test_arr = smt.fit_resample(input_test_arr, test_target_arr)
            logging.info(f"After Resampling - Test Data: {test_input_arr.shape} || Test Data: {target_test_arr.shape}")

            train_arr = np.c_[train_input_arr, target_train_arr]
            test_arr = np.c_[test_input_arr, target_test_arr]
            logging.info(f"Dataset after Joining with the Transformed Output: {train_arr.shape}")
            # logging.info(f"Saving the Processed Data: {pd.DataFrame(data=train_input_arr, columns=transformation_pipeline.feature_names_in_).head()}")
            utils.save_numpy_data(file_dir=self.data_transformation_config.transform_train_dir, array=train_arr)
            utils.save_numpy_data(file_dir=self.data_transformation_config.transform_test_dir, array=test_arr)

            logging.info("Saving the Preprocessor Object and Target Encoder")
            utils.save_object(file_dir=self.data_transformation_config.transform_obj_dir, obj=transformation_pipeline)
            utils.save_object(file_dir=self.data_transformation_config.target_encoder_dir, obj=label)

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