import os, sys 
from src.logger import logging 
from src.exception import CustomException
from src.components.model_resolver import ModelResolver
from src.utils import load_object, save_object
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelPusherArtifact, ModelTrainerArtifact, DataTransformationArtifacts

class ModelPusher: 

    def __init__(self, model_pusher_config:ModelPusherConfig, data_transformation_artifact:DataTransformationArtifacts, model_trainer_artifact: ModelTrainerArtifact):

        print(f"{'='*20}Model Pusher{'='*20}")
        self.model_pusher_config = model_pusher_config
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            transformer = load_object(file_dir=self.data_transformation_artifact.transform_obj_dir)
            model = load_object(file_dir=self.model_trainer_artifact.model_dir)
            target_encoder = load_object(file_dir=self.data_transformation_artifact.target_encoder_dir)

            save_object(file_dir=self.model_pusher_config.pusher_model_path, obj=model)
            save_object(file_dir=self.model_pusher_config.pusher_transformer_path, obj=transformer)
            save_object(file_dir=self.model_pusher_config.pusher_target_enc_path, obj=target_encoder)

            transformer_path = self.model_resolver.get_latest_save_transformer_path()
            model_path = self.model_resolver.get_latest_save_model_path()
            target_enc_path = self.model_resolver.get_latest_save_target_encoder_path()

            save_object(file_dir=transformer_path, obj=transformer)
            save_object(file_dir=model_path, obj=model)
            save_object(file_dir=target_enc_path, obj=target_encoder)

            model_pusher_artifact = ModelPusherArtifact(
                pusher_model_dir=self.model_pusher_config.model_pusher_dir,
                saved_model_dir=self.model_pusher_config.saved_model_dir
            )

            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys)