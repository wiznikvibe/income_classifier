import os, sys 
from typing import Optional 
from src.logger import logging 
from src.exception import CustomException
from src.entity.config_entity import TRANSFORMER_OBJ_FILE_NAME, MODEL_FILE_NAME, TARGET_ENCODER_OBJ_FILE_NAME

class ModelResolver:

    def __init__(
        self,
        model_registry: str = 'saved_models',
        transformer_dir_name = 'transformer',
        target_encoder_dir_name = 'target_encoder',
        model_dir_name = 'model'
    ):

        self.model_registry = model_registry
        os.makedirs(self.model_registry, exist_ok=True)
        self.transformer_dir_name = transformer_dir_name
        self.target_encoder_dir_name = target_encoder_dir_name
        self.model_dir_name = model_dir_name

    def get_latest_dir_path(self)-> Optional[str]:
        try:
            dir_names = os.listdir(self.model_registry)
            if len(dir_names) == 0:
                return None
            dir_names = list(map(int, dir_names))
            latest_dir_name = max(dir_names)
            return os.path.join(self.model_registry,f"{latest_dir_name}")
        
        except Exception as e:
            raise e

    def get_latest_model_path(self):
        try:
            latest_dir_path = get_latest_dir_path()
            if latest_dir_path is None:
                raise Exception(f"Model is not Available")
            return os.path.join(latest_dir_path, self.model_dir_name, MODEL_FILE_NAME)
        
        
        except Exception as e:
            raise e 


    def get_latest_transformer_path(self):
        try:
            latest_dir_path = get_latest_dir_path() 
            if latest_dir_path is None:
                raise Exception("Transformer is Not Created")
            return os.path.join(latest_dir_path, self.transformer_dir_name, TRANSFORMER_OBJ_FILE_NAME)
        except Exception as e:
            raise e

    def get_latest_target_encoder_path(self):
        try:
            latest_dir_path = get_latest_dir_path() 
            if latest_dir_path is None:
                raise Exception("Transformer is Not Created")
            return os.path.join(latest_dir_path, self.target_encoder_dir_name, TARGET_ENCODER_OBJ_FILE_NAME)
        except Exception as e:
            raise e 

    def get_latest_saved_dir_path(self):
        try:
            latest_dir_path = self.get_latest_dir_path()
            if latest_dir_path is None:
                return os.path.join(self.model_registry, f"{0}")
            latest_dir_num = int(os.path.basename(self.get_latest_dir_path()))
            return os.path.join(self.model_registry, f"{latest_dir_num+1}")
        except Exception as e:
            raise e 

    
    def get_latest_save_model_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir,self.model_dir_name,MODEL_FILE_NAME)
        except Exception as e:
            raise e

    def get_latest_save_transformer_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir,self.transformer_dir_name,TRANSFORMER_OBJECT_FILE_NAME)
        except Exception as e:
            raise e

    def get_latest_save_target_encoder_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir,self.target_encoder_dir_name,TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise e

