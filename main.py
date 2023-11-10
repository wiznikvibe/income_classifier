import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from src.pipeline.training_pipeline import start_training_pipeline 


if __name__ == '__main__':
    start_training_pipeline()
    
   