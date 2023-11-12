import os, sys 
import numpy as np 
import pandas as pd 
from dataclasses import dataclass 
from src.components.model_resolver import ModelResolver
from src.utils import load_object
from src.logger import logging 
from src.exception import CustomException

class PredictionPipeline:

    def __init__(self):
        self.model_resolver = ModelResolver()

    def predict(self, features):
        preprocessor_path = self.model_resolver.get_latest_transformer_path()
        print(preprocessor_path)
        model_path = self.model_resolver.get_latest_model_path()
        
        preprocessor = load_object(file_dir=preprocessor_path)
        # print(preprocessor.feature_names_in_)
        model = load_object(file_dir=model_path)

        scaled = preprocessor.transform(features)
        pred = model.predict(scaled) 
        
        return pred


class CustomClass:
    def __init__(self, age:int, fnlwgt:int, workclass:str, education:str, education_num:int, marital_status:str, occupation:str, relationship:str, race:str, sex:str, capital_gain:int, capital_loss:int, hours_per_week:int, native_country:str):
        # Current FEATURES:['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        
        self.age = age
        self.fnlwgt = fnlwgt
        self.workclass = workclass 
        self.education = education
        self.education_num = education_num
        self.marital_status = marital_status
        self.occupation = occupation 
        self.relationship = relationship
        self.race = race 
        self.sex = sex
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss 
        self.hours_per_week = hours_per_week
        self.native_country = native_country

    def get_data_DataFrame(self):
        try:
            custom_input = {
                'age':[self.age],
                'fnlwgt':[self.fnlwgt],
                'workclass':[self.workclass],
                'education':[self.education],
                'education-num':[self.education_num],
                'marital-status':[self.marital_status],
                'occupation':[self.occupation],
                'relationship':[self.relationship],
                'race':[self.race],
                'sex':[self.sex],
                'capital-gain':[self.capital_gain],
                'capital-loss':[self.capital_loss],
                'hours-per-week':[self.hours_per_week],
                'native-country':[self.native_country]

            } 

            data = pd.DataFrame(custom_input)
            return data
        except Exception as e:
            raise CustomException(e, sys)    
    