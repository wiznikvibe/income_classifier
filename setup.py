from setuptools import setup, find_packages
from typing import List



def get_requirements(file_path:str)-> List[str]:
    requirements = []
    HYPEN_DOT_E = "-e ."
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]
        if HYPEN_DOT_E in requirements:
            requirements.remove(HYPEN_DOT_E)

    
    return requirements


setup(
    name="income_classifier",
    version="0.0.1",
    description="Income Classification",
    author="Nikhil",
    author_email="nikhilshetty439@gmail.com",
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')
)