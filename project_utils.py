from dataclasses import dataclass
from dataclass_csv import DataclassReader, DataclassWriter

@dataclass
class Project:
    Project_name: str
    Zooniverse_number: str = None
    db_path: str = None
    server: str = None
    bucket: str = None
    key: str = None
    csv_folder: str = None
    movie_folder: str = None
    photo_folder: str = None


def find_project(project_path: str = "../db_starter/projects_list.csv", project_name: str = ''):
    '''Find project information using
       project csv path and project name'''
    with open(project_path) as csv:
        reader = DataclassReader(csv, Project)
        for row in reader:
            if row.Project_name == project_name:
                return row

def add_project(project_path: str = "../db_starter/projects_list.csv", project_info: dict = {}):
    '''Add new project information to
    project csv using a project_info dictionary
    '''
    with open(project_path, "a") as f:
        project = [Project(*list(project_info.values()))]
        w = DataclassWriter(f, project, Project)
        w.write(skip_header=True)

