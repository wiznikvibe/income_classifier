import os, sys 
from src.entity import config_entity, artifact_entity
from src.exception import CustomException
from src.logger import logging 
from src import utils 
from typing import Optional 
from sklearn.metrics import f1_score 
import pandas as pd 
import numpy as np 
from catboost import CatBoostClassifier


class ModelTrainer: 

    def __init__(self, model_trainer_config: config_entity.ModelTrainerConfig, data_transformation_artifact: artifact_entity.DataTransformationArtifacts):

        print(f"{'='*20} Model Trainer {'='*20}")
        logging.info(f"{'='*20} Model Trainer {'='*20}")

        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact


    def train_model(self, X, y):
        try:
            clf = CatBoostClassifier()
            clf.fit(X, y)
            return clf 
        except Exception as e:
            CustomException(e, sys)


    def initiate_model_trainer(self, )-> artifact_entity.ModelTrainerArtifact:
        try:
            logging.info("Loading the Transformed Data for Model Trainer")
            train_df = utils.load_numpy_obj(self.data_transformation_artifact.transform_train_dir)
            test_df = utils.load_numpy_obj(self.data_transformation_artifact.transform_test_dir)
            logging.info(f"Shape of Train Data: {train_df.shape} || Shape of Test Data: {test_df.shape} \n")

            logging.info("Splitting the Train and Test data ")
            X_train, y_train = train_df[:,:-1], train_df[:,-1]
            X_test, y_test = test_df[:,:-1], test_df[:,-1]

            model = self.train_model(X=X_train, y=y_train)
            
            logging.info("Computing th F1_score for Train Data")
            y_train_pred = model.predict(X_train)
            f1_train_score = f1_score(y_true=y_train, y_pred=y_train_pred)

            logging.info("Computing th F1_score for Test Data")
            y_test_pred = model.predict(X_test)
            f1_test_score = f1_score(y_true=y_test, y_pred=y_test_pred)

            logging.info(f"F1 Score: Train Data- {f1_train_score} || Test Data- {f1_test_score}")
            logging.info("Checking for Overfitting and Underfitting Conditions")

            if f1_test_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model Performance: Poor || Expected Accuracy over {self.model_trainer_config.expected_score*100}%")

            if abs(f1_train_score-f1_test_score) > self.model_trainer_config.overfitting_thresh:
                raise Exception(f"Difference {abs(f1_train_score-f1_test_score)} > Threshold: {self.model_trainer_config.overfitting_thresh}")

            utils.save_object(file_dir=self.model_trainer_config.model_dir, obj=model)
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                model_dir=self.model_trainer_config.model_dir,
                f1_train_score= f1_train_score,
                f1_test_score= f1_test_score
            )

            return model_trainer_artifact
                        
        except Exception as e:
            raise CustomException(e, sys)