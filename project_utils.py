# base imports
import os
import logging
import pandas as pd
from dataclasses import dataclass
from dataclass_csv import DataclassReader, DataclassWriter

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

@dataclass
class Project:
    Project_name: str
    Zooniverse_number: int = 0
    db_path: str = None
    server: str = None
    bucket: str = None
    key: str = None
    csv_folder: str = None
    movie_folder: str = None
    photo_folder: str = None


def find_project(project_name: str = ''):
    '''Find project information using
       project csv path and project name'''
    # Specify the path to the list of projects
    project_path = "../db_starter/projects_list.csv"
    snic_path = "/cephyr/NOBACKUP/groups/snic2021-6-9/"
        
    # Check path to the list of projects is a csv
    if os.path.exists(project_path) and not project_path.endswith(".csv"):
        logging.error("A csv file was not selected. Please try again.")

    elif os.path.exists(project_path) and os.path.exists(snic_path):
        project_path = os.path.join(snic_path, "db_starter/projects_list.csv")
        
    # If list of projects doesn't exist retrieve it from github
    elif not os.path.exists(project_path):
        github_path = "https://github.com/ocean-data-factory-sweden/koster_data_management/blob/main/db_starter/projects_list.csv?raw=true"
        read_file = pd.read_csv(github_path)
        read_file.to_csv(project_path, index=None)

    with open(project_path) as csv:
        reader = DataclassReader(csv, Project)
        for row in reader:
            if row.Project_name == project_name:
                return row

def add_project(project_info: dict = {}):
    '''Add new project information to
    project csv using a project_info dictionary
    '''
    project_path = "../db_starter/projects_list.csv"
    snic_path = "/cephyr/NOBACKUP/groups/snic2021-6-9/"

    if not os.path.exists(project_path) and os.path.exists(snic_path):
        project_path = os.path.join(snic_path, "db_starter/projects_list.csv")
    with open(project_path, "a") as f:
        project = [Project(*list(project_info.values()))]
        w = DataclassWriter(f, project, Project)
        w.write(skip_header=True)

