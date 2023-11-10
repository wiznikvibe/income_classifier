from src.components.model_resolver import ModelResolver
from src.entity import config_entity, artifact_entity
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from sklearn.metrics import f1_score 
import os, sys 
import pandas as pd 

class ModelEvaluation:

    def __init__(self, model_eval_config:config_entity.ModelEvaluationConfig, data_ingestion_artifact:artifact_entity.DataIngestionArtifact, data_transformation_artifact:artifact_entity.DataTransformationArtifacts, model_trainer_artifact: artifact_entity.ModelTrainerArtifact):

        print(f"{'='*20}Model Evaluation{'='*20}")
        self.model_eval_config = model_eval_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self.model_resolver = ModelResolver()

    def initiate_model_evaluation(self)-> artifact_entity.ModelEvaluationArtifact:
        try:
            logging.info(f"{'='*20}Model Evaluation Initiated{'='*20}")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            
            if latest_dir_path is None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, improved_acc=None)
                logging.info(f"{model_eval_artifact}")
                return model_eval_artifact
            
            logging.info("Loading Model, Transformer and Target_Encoder for the Saved Model.")
            model_path = self.model_resolver.get_latest_model_path()
            transformer_path = self.model_resolver.get_latest_transformer_path()
            target_enc_path = self.model_resolver.get_latest_target_encoder_path()

            model = load_object(file_dir=model_path)
            transformer = load_object(file_dir=transformer_path)
            target_encoder = load_object(file_dir=target_enc_path)

            current_model = load_object(file_dir=self.model_trainer_artifact.model_dir)
            current_transformer = load_object(file_dir=self.data_transformation_artifact.transform_obj_dir)
            current_target_encoder = load_object(file_dir=self.data_transformation_artifact.target_encoder_dir)

            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_dir)
            target_df = test_df[self.model_eval_config.target_column]
            y_true = target_encoder.transform(target_df)

            logging.info("Accuracy Test: Model in Saved Model")
            input_feature_names = list(transformer.feature_names_in_)
            logging.info(f"FEATURES:{input_feature_names}")
            input_arr = transformer.transform(test_df[input_feature_names])
            y_pred = model.predict(input_arr)
            saved_model_score = f1_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy Score for Saved Model: {saved_model_score}")

            logging.info("Accuracy Test for Model in Testnet")
            input_feature_names = list(current_transformer.feature_names_in_)
            logging.info(f"Current FEATURES:{input_feature_names}")

            y_true = current_target_encoder.transform(target_df)
            input_arr = current_transformer.transform(test_df[input_feature_names])
            y_pred = current_model.predict(input_arr)
            current_model_score = f1_score(y_pred=y_pred, y_true=y_true)
            logging.info(f"Accuracy Score for Model in Testnet: {current_model_score}")

            if current_model_score <= saved_model_score:
                logging.info(f"Model in Testnet: Not Improved")
                raise Exception("Accuracy not IMPROVED")
            
            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                is_model_accepted=True,
                improved_acc= current_model_score-saved_model_score
            )

            return model_eval_artifact

        except Exception as e:
            raise CustomException(e, sys)

