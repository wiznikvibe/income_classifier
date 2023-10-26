import os, sys 
from pathlib import Path
import logging 

logging.basicConfig(level=logging.INFO)
while True:
    project_name = input("Project Folder Name: ")
    if project_name != "":
        break

crucks = [
    f"{project_name}/__init__.py",
    f"{project_name}/exception.py",
    f"{project_name}/logger.py",
    f"{project_name}/utils.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/model_pusher.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/pipeline/prediction_pipeline.py",
    "app.py",
    "Dockerfile",
    "setup.py"
]

for component in crucks:
    file_path = Path(component)
    file_dir, filename = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Creating Directory {file_dir} / file: {filename}")

    if not(os.path.exists(file_path)) or (os.path.getsize(file_path)==0):
        with open(file_path, "w") as f:
            pass 

    else:
        logging.info("File Structure Already Initiated")
    