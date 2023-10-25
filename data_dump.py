import json
import pandas as pd 
from src.config import mongo_client

DATA_FILE_PATH = r"C:/Users/nikhi/income_classifier/dataset/salary.csv"
DATABASE_NAME = 'hr'
COLLECTION_NAME = 'details'

if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    df.reset_index(drop=True, inplace=True)

    json_records = list(json.loads(df.T.to_json()).values())
    print(json_records[0])

    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_records)
    print(f"{'='*20} DATA UPLOADING COMPLETE {'='*20}")