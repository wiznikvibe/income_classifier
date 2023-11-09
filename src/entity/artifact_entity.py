from dataclasses import dataclass

@dataclass 
class DataIngestionArtifact:
    feature_store_dir: str
    train_file_dir: str
    test_file_dir: str

@dataclass 
class DataValidationArtifact:
    report_file_dir: str 

@dataclass
class DataTransformationArtifacts:
    transform_obj_dir: str
    transform_train_dir: str
    transform_test_dir: str
    target_encoder_dir: str

@dataclass 
class ModelTrainerArtifact:
    model_dir: str 
    f1_train_score: float 
    f1_test_score: float 

@dataclass 
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_acc: float 

@dataclass 
class ModelPusherArtifact:
    pusher_model_dir: str 
    saved_model_dir: str 
    