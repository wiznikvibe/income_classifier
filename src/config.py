import os, sys
import pymongo 
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass 
class EnviromentVariable:
    mongo_db_url:str = os.getenv("MONGO_DB_URL")


env_var = EnviromentVariable()
mongo_client = pymongo.MongoClient(env_var.mongo_db_url)
