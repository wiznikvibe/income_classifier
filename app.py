from src.entity import config_entity, artifact_entity
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation


if __name__ == '__main__':
    training_pipeline_config = config_entity.TrainingPipelineConfig()
    data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)

    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
    
    data_validation_config = config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
    data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
    data_validation_artifact = data_validation.initiate_data_validation()
    print(data_validation_artifact)
    
   