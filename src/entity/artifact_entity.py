from dataclasses import dataclass

@dataclass 
class DataIngestionArtifact:
    feature_store_dir:str
    train_file_dir:str
    test_file_dir:str

@dataclass 
class DataValidationArtifact:
    report_file_dir: str 

    