# base imports
import os
import logging
import pandas as pd
from dataclasses import dataclass
from dataclass_csv import DataclassReader, DataclassWriter, exceptions

# util imports
import kso_utils.spyfish_utils as spyfish_utils


# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

# Specify volume allocated by SNIC
snic_path = "/mimer/NOBACKUP/groups/snic2022-22-1210"


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
    ml_folder: str = None


def find_project(project_name: str = ""):
    """Find project information using
    project csv path and project name"""
    # Specify the path to the list of projects
    project_path = "../kso_utils/db_starter/projects_list.csv"

    # Check path to the list of projects is a csv
    if os.path.exists(project_path) and not project_path.endswith(".csv"):
        logging.error("A csv file was not selected. Please try again.")

    # If list of projects doesn't exist retrieve it from github
    elif not os.path.exists(project_path):
        github_path = "https://github.com/ocean-data-factory-sweden/kso_utils/blob/main/db_starter/projects_list.csv?raw=true"
        read_file = pd.read_csv(github_path)
        read_file.to_csv(project_path, index=None)

    with open(project_path) as csv:
        reader = DataclassReader(csv, Project)
        try:
            for row in reader:
                if row.Project_name == project_name:
                    logging.info(f"{project_name} loaded succesfully")
                    return row
        except exceptions.CsvValueError:
            logging.error(
                f"This project {project_name} does not contain any csv information. Please select another."
            )


def add_project(project_info: dict = {}):
    """Add new project information to
    project csv using a project_info dictionary
    """
    project_path = "../kso_utils/db_starter/projects_list.csv"

    if not os.path.exists(project_path) and os.path.exists(snic_path):
        project_path = os.path.join(snic_path, "db_starter/projects_list.csv")
    with open(project_path, "a") as f:
        project = [Project(*list(project_info.values()))]
        w = DataclassWriter(f, project, Project)
        w.write(skip_header=True)


def get_col_names(project: Project, local_csv: str):
    """Return a dictionary with the project-specific column names of a csv of interest
    This function helps matching the schema format without modifying the column names of the original csv.

    :param project: The project object
    :param local_csv: a string of the name of the local csv of interest
    :return: a dictionary with the names of the columns
    """

    # Get project-specific server info
    project_name = project.Project_name

    if "sites" in local_csv:
        # Get spyfish specific column names
        if project_name == "Spyfish_Aotearoa":
            col_names_sites = spyfish_utils.get_spyfish_col_names("sites")

        else:
            # Save the column names of interest in a dict
            col_names_sites = {
                "siteName": "siteName",
                "decimalLatitude": "decimalLatitude",
                "decimalLongitude": "decimalLongitude",
                "geodeticDatum": "geodeticDatum",
                "countryCode": "countryCode",
            }

        return col_names_sites

    if "movies" in local_csv:
        # Get spyfish specific column names
        if project_name == "Spyfish_Aotearoa":
            col_names_movies = spyfish_utils.get_spyfish_col_names("movies")

        elif project_name == "Koster_Seafloor_Obs":
            # Save the column names of interest in a dict
            col_names_movies = {
                "filename": "filename",
                "created_on": "created_on",
                "fps": "fps",
                "duration": "duration",
                "sampling_start": "SamplingStart",
                "sampling_end": "SamplingEnd",
                "author": "Author",
                "site_id": "site_id",
                "fpath": "fpath",
            }

        else:
            # Save the column names of interest in a dict
            col_names_movies = {
                "filename": "filename",
                "created_on": "created_on",
                "fps": "fps",
                "duration": "duration",
                "sampling_start": "sampling_start",
                "sampling_end": "sampling_end",
                "author": "author",
                "site_id": "site_id",
                "fpath": "fpath",
            }

        return col_names_movies

    if "species" in local_csv:
        # Save the column names of interest in a dict
        col_names_species = {
            "label": "label",
            "scientificName": "scientificName",
            "taxonRank": "taxonRank",
            "kingdom": "kingdom",
        }
        return col_names_species

    else:
        raise ValueError("The local csv doesn't have a table match in the schema")
