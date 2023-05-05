# base imports
import os
import glob
import math
import json
import random
import logging
import asyncio
import wandb
import folium
import subprocess
import numpy as np
import pandas as pd
import ipywidgets as widgets
import ffmpeg
import shutil
import datetime
import paramiko
from paramiko import SSHClient
from scp import SCPClient
from itertools import chain
from pathlib import Path
from tqdm import tqdm
import imagesize
import ipysheet
import multiprocessing
from IPython.display import display, HTML, clear_output
from ipyfilechooser import FileChooser

# Zooniverse imports
from panoptes_client import (
    SubjectSet,
    Subject,
)

# util imports
import kso_utils.tutorials_utils as t_utils
import kso_utils.project_utils as project_utils
import kso_utils.db_utils as db_utils
import kso_utils.movie_utils as movie_utils
import kso_utils.server_utils as server_utils
import kso_utils.yolo_utils as yolo_utils
import kso_utils.zooniverse_utils as zu_utils

# project-specific imports
from kso_utils.koster_utils import (
    process_koster_movies_csv,
    filter_bboxes,
    process_clips_koster,
)
from kso_utils.sgu_utils import process_sgu_photos_csv
from kso_utils.spyfish_utils import (
    process_spyfish_movies,
    process_spyfish_sites,
    process_clips_spyfish,
    check_spyfish_movies,
    get_spyfish_choices,
    get_spyfish_col_names,
    spyfish_subject_metadata,
)

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


# General project utilities
def import_model_modules(module_names):
    importlib = __import__("importlib")
    modules = {}
    for module_name, module_full in zip(["train", "detect", "val"], module_names):
        try:
            modules[module_name] = importlib.import_module(module_full)
        except ModuleNotFoundError:
            logging.error(f"Module {module_name} could not be imported.")
    return modules


def import_modules(module_names, utils: bool = True, models: bool = False):
    importlib = __import__("importlib")
    modules = {}
    model_presets = ["train", "detect", "val"]
    for i, module_name in enumerate(module_names):
        if utils:
            module_full = "kso_utils." + module_name
        else:
            module_full = module_name
        try:
            if models:
                module_name = model_presets[i]
            modules[module_name] = importlib.import_module(module_full)
        except ModuleNotFoundError:
            logging.error(f"Module {module_name} could not be imported.")
    return modules


def parallel_map(func, iterable, args=()):
    with multiprocessing.Pool() as pool:
        results = pool.starmap(func, zip(iterable, *args))
    return results


class ProjectProcessor:
    def __init__(self, project: project_utils.Project):
        self.project = project
        self.db_connection = None
        self.server_info = {}
        self.db_info = {}
        self.zoo_info = {}
        self.annotation_engine = None
        self.annotations = pd.DataFrame()
        self.classifications = pd.DataFrame()
        self.generated_clips = pd.DataFrame()

        # Import modules
        self.modules = import_modules([])
        # Create empty meta tables
        self.init_meta()
        # Setup initial db
        self.setup_db()
        # Get server details from the db_info
        self.server_info = {
            x: self.db_info[x]
            for x in ["client", "sftp_client"]
            if x in self.db_info.keys()
        }
        if self.project.movie_folder is not None:
            # Check movies on server
            self.get_movie_info()
        # Reads csv files
        self.load_meta()

    def __repr__(self):
        return repr(self.__dict__)

    def keys(self):
        """Print keys of ProjectProcessor object"""
        logging.info("Stored variable names.")
        return list(self.__dict__.keys())

    # general
    def mount_snic(self, snic_path: str = "/mimer/NOBACKUP/groups/snic2021-6-9/"):
        """
        It mounts the remote directory to the local machine

        :param snic_path: The path to the SNIC directory on the remote server, defaults to
               /mimer/NOBACKUP/groups/snic2021-6-9/
        :type snic_path: str (optional)
        :return: The return value is the exit status of the command.
        """
        cmd = "sshfs {}:{} {}".format(
            self.server_info["client"].get_transport().get_username(),
            snic_path,
            snic_path,
        )
        stdin, stdout, stderr = self.server_info["client"].exec_command(cmd)
        # Print output and errors (if any)
        logging.info("Output:", stdout.read().decode("utf-8"))
        logging.error("Errors:", stderr.read().decode("utf-8"))
        # Verify that the remote directory is mounted
        if os.path.ismount(snic_path):
            logging.info("Remote directory mounted successfully!")
            return 1
        else:
            logging.error("Failed to mount remote directory!")
            return 0

    def setup_db(self):
        """
        The function checks if the project is running on the SNIC server, if not it attempts to mount
        the server. If the server is available, the function creates a database and adds the database to
        the project
        :return: The database connection object.
        """
        if self.project.server == "SNIC":
            if not os.path.exists(self.project.csv_folder):
                logging.error("Not running on SNIC server, attempting to mount...")
                status = self.mount_snic()
                if status == 0:
                    return
        db_utils.init_db(self.project.db_path)
        self.db_info = self.initiate_db()
        # connect to the database and add to project
        self.db_connection = db_utils.create_connection(self.project.db_path)

    def initiate_db(self):
        """
        This function takes a project name as input and returns a dictionary with all the information needed
        to connect to the project's database

        :param project: The name of the project. This is used to get the project-specific info from the config file
        :return: A dictionary with the following keys (db_path, project_name, server_i_dict, db_initial_info)

        """

        # Check if template project
        if self.project.Project_name == "model-registry":
            return {}

        # Get project specific info
        server_i_dict, db_initial_info = self.get_project_details()

        # Check if server and db info
        if len(server_i_dict) == 0 and len(db_initial_info) == 0:
            return {}

        # Initiate the sql db
        db_utils.init_db(db_initial_info["db_path"])

        # List the csv files of interest
        list_of_init_csv = [
            "local_sites_csv",
            "local_movies_csv",
            "local_photos_csv",
            "local_species_csv",
        ]

        # Populate the sites, movies, photos, info
        for local_i_csv in list_of_init_csv:
            if local_i_csv in db_initial_info.keys():
                self.populate_db(db_initial_info=db_initial_info, local_csv=local_i_csv)

        # Combine server/project info in a dictionary
        db_info_dict = {**db_initial_info, **server_i_dict}

        return db_info_dict

    def populate_db(self, db_initial_info: dict, local_csv: str):
        """
        > This function populates a sql table of interest based on the info from the respective csv

        :param db_initial_info: The dictionary containing the initial database information
        :param project: The project object
        :param local_csv: a string of the names of the local csv to populate from
        """

        # Process the csv of interest and tests for compatibility with sql table
        csv_i, df = self.process_test_csv(
            db_info_dict=db_initial_info, local_csv=local_csv
        )

        # Add values of the processed csv to the sql table of interest
        db_utils.add_to_table(
            db_initial_info["db_path"],
            csv_i,
            [tuple(i) for i in df.values],
            len(df.columns),
        )

    def process_test_csv(self, db_info_dict: dict, local_csv: str):
        """
        > This function process a csv of interest and tests for compatibility with the respective sql table of interest

        :param db_info_dict: The dictionary containing the database information
        :param project: The project object
        :param local_csv: a string of the names of the local csv to populate from
        :return: a string of the category of interest and the processed dataframe

        """
        # Load the csv with the information of interest
        df = pd.read_csv(db_info_dict[local_csv])

        # Save the category of interest and process the df
        if "sites" in local_csv:
            field_names, csv_i, df = self.process_sites_df(df)

        if "movies" in local_csv:
            field_names, csv_i, df = self.process_movies_df(db_info_dict, df)

        if "species" in local_csv:
            field_names, csv_i, df = self.process_species_df(df)

        if "photos" in local_csv:
            field_names, csv_i, df = self.process_photos_df(db_info_dict, df)

        # Add the names of the basic columns in the sql db
        field_names = field_names + db_utils.get_column_names_db(db_info_dict, csv_i)
        field_names.remove("id")

        # Select relevant fields
        df.rename(columns={"Author": "author"}, inplace=True)
        df = df[[c for c in field_names if c in df.columns]]

        # Roadblock to prevent empty rows in id_columns
        db_utils.test_table(df, csv_i, [df.columns[0]])

        return csv_i, df

    def process_sites_df(
        self,
        df: pd.DataFrame,
    ):
        """
        > This function processes the sites dataframe and returns a string with the category of interest

        :param df: a pandas dataframe of the information of interest
        :return: a string of the category of interest and the processed dataframe
        """

        # Check if the project is the Spyfish Aotearoa
        if self.project.Project_name == "Spyfish_Aotearoa":
            # Rename columns to match schema fields
            df = process_spyfish_sites(df)

        # Specify the category of interest
        csv_i = "sites"

        # Specify the id of the df of interest
        field_names = ["site_id"]

        return field_names, csv_i, df

    def process_movies_df(self, db_info_dict: dict, df: pd.DataFrame):
        """
        > This function processes the movies dataframe and returns a string with the category of interest

        :param db_info_dict: The dictionary containing the database information
        :param df: a pandas dataframe of the information of interest
        :return: a string of the category of interest and the processed dataframe
        """

        # Check if the project is the Spyfish Aotearoa
        if self.project.Project_name == "Spyfish_Aotearoa":
            df = process_spyfish_movies(df)

        # Check if the project is the KSO
        if self.project.Project_name == "Koster_Seafloor_Obs":
            df = process_koster_movies_csv(df)

        # Connect to database
        conn = db_utils.create_connection(db_info_dict["db_path"])

        # Reference movies with their respective sites
        sites_df = pd.read_sql_query("SELECT id, siteName FROM sites", conn)
        sites_df = sites_df.rename(columns={"id": "site_id"})

        # Merge movies and sites dfs
        df = pd.merge(df, sites_df, how="left", on="siteName")

        # Select only those fields of interest
        if "fpath" not in df.columns:
            df["fpath"] = df["filename"]

        # Specify the category of interest
        csv_i = "movies"

        # Specify the id of the df of interest
        field_names = ["movie_id"]

        return field_names, csv_i, df

    def process_photos_df(self, db_info_dict: dict, df: pd.DataFrame):
        """
        > This function processes the photos dataframe and returns a string with the category of interest

        :param db_info_dict: The dictionary containing the database information
        :param df: a pandas dataframe of the information of interest
        :param project: The project object
        :return: a string of the category of interest and the processed dataframe
        """
        # Check if the project is the SGU
        if self.project.Project_name == "SGU":
            df = process_sgu_photos_csv(db_info_dict)

        # Specify the category of interest
        csv_i = "photos"

        # Specify the id of the df of interest
        field_names = ["ID"]

        return field_names, csv_i, df

    def process_species_df(self, df: pd.DataFrame):
        """
        > This function processes the species dataframe and returns a string with the category of interest

        :param df: a pandas dataframe of the information of interest
        :param project: The project object
        :return: a string of the category of interest and the processed dataframe
        """

        # Rename columns to match sql fields
        df = df.rename(columns={"commonName": "label"})

        # Specify the category of interest
        csv_i = "species"

        # Specify the id of the df of interest
        field_names = ["species_id"]

        return field_names, csv_i, df

    def get_db_init_info(self, server_dict: dict) -> dict:
        """
        This function downloads the csv files from the server and returns a dictionary with the paths to the
        csv files

        :param project: the project object
        :param server_dict: a dictionary containing the server information
        :type server_dict: dict
        :return: A dictionary with the paths to the csv files with the initial info to build the db.
        """

        # Define the path to the csv files with initial info to build the db
        db_csv_info = self.project.csv_folder

        # Get project-specific server info
        server = self.project.server
        project_name = self.project.Project_name

        # Create the folder to store the csv files if not exist
        if not os.path.exists(db_csv_info):
            Path(db_csv_info).mkdir(parents=True, exist_ok=True)
            # Recursively add permissions to folders created
            [os.chmod(root, 0o777) for root, dirs, files in os.walk(db_csv_info)]

        if server == "AWS":
            # Download csv files from AWS
            db_initial_info = server_utils.download_csv_aws(
                project_name, server_dict, db_csv_info
            )

            if project_name == "Spyfish_Aotearoa":
                db_initial_info = get_spyfish_choices(
                    server_dict, db_initial_info, db_csv_info
                )

        elif server in ["LOCAL", "SNIC"]:
            csv_folder = db_csv_info

            # Define the path to the csv files with initial info to build the db
            if not os.path.exists(csv_folder):
                logging.error(
                    "Invalid csv folder specified, please provide the path to the species, sites and movies (optional)"
                )

            for file in Path(csv_folder).rglob("*.csv"):
                if "sites" in file.name:
                    sites_csv = file
                if "movies" in file.name:
                    movies_csv = file
                if "photos" in file.name:
                    photos_csv = file
                if "survey" in file.name:
                    surveys_csv = file
                if "species" in file.name:
                    species_csv = file

            if (
                "movies_csv" not in vars()
                and "photos_csv" not in vars()
                and os.path.exists(csv_folder)
            ):
                logging.info(
                    "No movies or photos found, an empty movies file will be created."
                )
                with open(str(Path(f"{csv_folder}", "movies.csv")), "w") as fp:
                    fp.close()

            db_initial_info = {}

            if "sites_csv" in vars():
                db_initial_info["local_sites_csv"] = sites_csv

            if "species_csv" in vars():
                db_initial_info["local_species_csv"] = species_csv

            if "movies_csv" in vars():
                db_initial_info["local_movies_csv"] = movies_csv

            if "photos_csv" in vars():
                db_initial_info["local_photos_csv"] = photos_csv

            if "surveys_csv" in vars():
                db_initial_info["local_surveys_csv"] = surveys_csv

            if len(db_initial_info) == 0:
                logging.error(
                    "Insufficient information to build the database. Please fix the path to csv files."
                )

        elif server == "TEMPLATE":
            # Specify the id of the folder with csv files of the template project
            gdrive_id = "1PZGRoSY_UpyLfMhRphMUMwDXw4yx1_Fn"

            # Download template csv files from Gdrive
            db_initial_info = server_utils.download_gdrive(gdrive_id, db_csv_info)

            for file in Path(db_csv_info).rglob("*.csv"):
                if "sites" in file.name:
                    sites_csv = file
                if "movies" in file.name:
                    movies_csv = file
                if "photos" in file.name:
                    photos_csv = file
                if "survey" in file.name:
                    surveys_csv = file
                if "species" in file.name:
                    species_csv = file

            db_initial_info = {}

            if "sites_csv" in vars():
                db_initial_info["local_sites_csv"] = sites_csv

            if "species_csv" in vars():
                db_initial_info["local_species_csv"] = species_csv

            if "movies_csv" in vars():
                db_initial_info["local_movies_csv"] = movies_csv

            if "photos_csv" in vars():
                db_initial_info["local_photos_csv"] = photos_csv

            if "surveys_csv" in vars():
                db_initial_info["local_surveys_csv"] = surveys_csv

            if len(db_initial_info) == 0:
                logging.error(
                    "Insufficient information to build the database. Please fix the path to csv files."
                )

        else:
            raise ValueError(
                "The server type you have chosen is not currently supported. Supported values are AWS, SNIC and LOCAL."
            )

        # Add project-specific db_path
        db_initial_info["db_path"] = self.project.db_path
        if "client" in server_dict:
            db_initial_info["client"] = server_dict["client"]

        return db_initial_info

    def get_project_details(self):
        """
        > This function connects to the server (or folder) hosting the csv files, and gets the initial info
        from the database

        :param project: the project object
        """

        # Connect to the server (or folder) hosting the csv files
        server_i_dict = server_utils.connect_to_server(self.project)

        # Get the initial info
        db_initial_info = self.get_db_init_info(server_i_dict)

        return server_i_dict, db_initial_info

    def get_db_table(self, table_name, interactive: bool = False):
        """
        It takes a table name as an argument, connects to the database, gets the column names, gets the
        data, and returns a DataFrame or an interactive view of the table using HTML.

        :param table_name: The name of the table you want to get from the database
        :param interactive: A boolean which displays the table as HTML
        :return: A dataframe
        """
        cursor = self.db_connection.cursor()
        # Get column names
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

        # Get column names
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        columns = [col[1] for col in cursor.fetchall()]

        # Create a DataFrame from the data
        df = pd.DataFrame(rows, columns=columns)

        if interactive:
            html = f"<div style='height:300px;overflow:auto'>{df.to_html(index=False)}</div>"

            # Display the HTML
            display(HTML(html))
        else:
            return df

    def get_zoo_info(self, generate_export: bool = False):
        """
        It connects to the Zooniverse project, and then retrieves and populates the Zooniverse info for
        the project
        :return: The zoo_info is being returned.
        """
        if self.project.Zooniverse_number is not None:
            self.zoo_project = zu_utils.connect_zoo_project(self.project)
            self.zoo_info = zu_utils.retrieve__populate_zoo_info(
                self.project,
                self.db_info,
                self.zoo_project,
                zoo_info=["subjects", "workflows", "classifications"],
                generate_export=generate_export,
            )
        else:
            logging.error("This project is not registered with ZU.")
            return

    def get_movie_info(self):
        """
        It retrieves a csv file from the server, and then updates the local variable server_movies_csv
        with the contents of that csv file
        """
        self.server_movies_csv = movie_utils.retrieve_movie_info_from_server(
            self.project, self.db_info
        )
        logging.info("server_movies_csv updated")

    def load_movie(self, filepath):
        """
        It takes a filepath, and returns a movie path

        :param filepath: The path to the movie file
        :return: The movie path.
        """
        return movie_utils.get_movie_path(filepath, self.db_info, self.project)

    # t1
    def init_meta(self, init_keys=["movies", "species", "sites"]):
        """
        This function creates a new attribute for the class, which is a pandas dataframe.
        The attribute name is a concatenation of the string 'local_' and the value of the variable
        meta_name, and the string '_csv'.

        The value of the attribute is a pandas dataframe.

        The function is called with the argument init_keys, which is a list of strings.

        The function loops through the list of strings, and for each string, it creates a new attribute
        for the class.

        :param init_keys: a list of strings that are the names of the metadata files you want to
               initialize
        """
        for meta_name in init_keys:
            setattr(self, "local_" + meta_name + "_csv", pd.DataFrame())
            setattr(self, "server_" + meta_name + "_csv", pd.DataFrame())

    def load_meta(self, base_keys=["movies", "species", "sites"]):
        """
        It loads the metadata from the local csv files into the `db_info` dictionary

        :param base_keys: the base keys to load
        """
        for key, val in self.db_info.items():
            if any("local_" + ext in key for ext in base_keys):
                setattr(self, key, pd.read_csv(val))

    def select_meta_range(self, meta_key: str):
        """
        > This function takes a meta key as input and returns a dataframe, range of rows, and range of
        columns

        :param meta_key: str
        :type meta_key: str
        :return: meta_df, range_rows, range_columns
        """
        meta_df, range_rows, range_columns = t_utils.select_sheet_range(
            db_info_dict=self.db_info, orig_csv=f"local_{meta_key}_csv"
        )
        return meta_df, range_rows, range_columns

    def edit_meta(self, meta_df: pd.DataFrame, range_rows, range_columns):
        """
        > This function opens a Google Sheet with the dataframe passed as an argument

        :param meta_df: the dataframe that contains the metadata
        :type meta_df: pd.DataFrame
        :param range_rows: a list of row numbers to include in the sheet
        :param range_columns: a list of columns to display in the sheet
        :return: df_filtered, sheet
        """
        df_filtered, sheet = t_utils.open_csv(
            df=meta_df, df_range_rows=range_rows, df_range_columns=range_columns
        )
        display(sheet)
        return df_filtered, sheet

    def view_meta_changes(self, df_filtered, sheet):
        """
        > This function takes a dataframe and a sheet name as input, and returns a dataframe with the
        changes highlighted

        :param df_filtered: a dataframe that has been filtered by the user
        :param sheet: the name of the sheet you want to view
        :return: A dataframe with the changes highlighted.
        """
        highlight_changes, sheet_df = t_utils.display_changes(
            self.db_info, isheet=sheet, df_filtered=df_filtered
        )
        display(highlight_changes)
        return sheet_df

    def update_csv(
        self,
        sheet_df: pd.DataFrame,
        df: pd.DataFrame,
        local_csv: str,
        serv_csv: str,
    ):
        """
        This function is used to update the csv files locally and in the server

        :param db_info_dict: The dictionary containing the database information
        :param project: The project object
        :param sheet_df: The dataframe of the sheet you want to update
        :param df: a pandas dataframe of the information of interest
        :param local_csv: a string of the names of the local csv to update
        :param serv_csv: a string of the names of the server csv to update
        """
        # Create button to confirm changes
        confirm_button = widgets.Button(
            description="Yes, details are correct",
            layout=widgets.Layout(width="25%"),
            style={"description_width": "initial"},
            button_style="danger",
        )

        # Create button to deny changes
        deny_button = widgets.Button(
            description="No, I will go back and fix them",
            layout=widgets.Layout(width="45%"),
            style={"description_width": "initial"},
            button_style="danger",
        )

        # Save changes in survey csv locally and in the server
        async def f(sheet_df, df, local_csv, serv_csv):
            x = await t_utils.wait_for_change(
                confirm_button, deny_button
            )  # <---- Pass both buttons into the function
            if (
                x == "Yes, details are correct"
            ):  # <--- use if statement to trigger different events for the two buttons
                logging.info("Checking if changes can be incorporated to the database")

                # Retrieve the column name of the id of interest (Sites, movies,..)
                id_col = [col for col in df.columns if "_id" in col][0]

                # Replace the different values based on id
                df_orig = df.copy()
                df_new = sheet_df.copy()
                df_orig.set_index(id_col, inplace=True)
                df_new.set_index(id_col, inplace=True)
                df_orig.update(df_new)
                df_orig.reset_index(drop=False, inplace=True)

                # Process the csv of interest and tests for compatibility with sql table
                csv_i, df_to_db = self.process_test_csv(
                    project=self.project, local_csv=local_csv
                )

                # Log changes locally
                self.log_meta_changes(
                    meta_key=local_csv,
                    new_sheet_df=sheet_df,
                )

                # Save the updated df locally
                df_orig.to_csv(self.db_info[local_csv], index=False)
                logging.info("The local csv file has been updated")

                if self.project.server == "AWS":
                    # Save the updated df in the server
                    self.update_csv_server(orig_csv=serv_csv, updated_csv=local_csv)

            else:
                logging.info("Run this cell again when the changes are correct!")

        print("")
        print("Are the changes above correct?")
        display(
            widgets.HBox([confirm_button, deny_button])
        )  # <----Display both buttons in an HBox
        asyncio.create_task(f(sheet_df, df, local_csv, serv_csv))

    def update_meta(self, new_table, meta_name):
        """
        `update_meta` takes a new table, a meta name, and updates the local and server meta files

        :param new_table: the name of the table that you want to update
        :param meta_name: the name of the metadata file (e.g. "movies")
        :return: The return value is a boolean.
        """
        return self.update_csv(
            new_table,
            getattr(self, "local_" + meta_name + "_csv"),
            "local_" + meta_name + "_csv",
            "server_" + meta_name + "_csv",
        )

    def map_sites(self):
        """
        > This function takes a dictionary of database information and a project object as input, and
        returns a map of the sites in the database

        :param db_info_dict: a dictionary containing the information needed to connect to the database
        :type db_info_dict: dict
        :param project: The project object
        :return: A map with all the sites plotted on it.
        """
        if self.project.server in ["SNIC", "LOCAL"]:
            # Set initial location to Gothenburg
            init_location = [57.708870, 11.974560]

        else:
            # Set initial location to Taranaki
            init_location = [-39.296109, 174.063916]

        # Create the initial kso map
        kso_map = folium.Map(location=init_location, width=900, height=600)

        # Read the csv file with site information
        sites_df = pd.read_csv(self.db_info["local_sites_csv"])

        # Combine information of interest into a list to display for each site
        sites_df["site_info"] = sites_df.values.tolist()

        # Save the names of the columns
        df_cols = sites_df.columns

        # Add each site to the map
        sites_df.apply(
            lambda row: folium.CircleMarker(
                location=[
                    row[df_cols.str.contains("Latitude")],
                    row[df_cols.str.contains("Longitude")],
                ],
                radius=14,
                popup=row["site_info"],
                tooltip=row[df_cols.str.contains("siteName", case=False)],
            ).add_to(kso_map),
            axis=1,
        )

        # Add a minimap to the corner for reference
        kso_map = kso_map.add_child(folium.plugins.MiniMap())

        # Return the map
        return kso_map

    def preview_media(self):
        """
        > The function `preview_media` is a function that takes in a `self` argument and returns a
        function `f` that takes in three arguments: `project`, `db_info`, and `server_movies_csv`. The
        function `f` is an asynchronous function that takes in the value of the `movie_selected` widget
        and displays the movie preview
        """
        movie_selected = t_utils.select_movie(self.server_movies_csv)

        async def f(project, db_info, server_movies_csv):
            x = await t_utils.single_wait_for_change(movie_selected, "value")
            html, movie_path = movie_utils.preview_movie(
                project, db_info, server_movies_csv, x
            )
            display(html)
            self.movie_selected = x
            self.movie_path = movie_path

        asyncio.create_task(f(self.project, self.db_info, self.server_movies_csv))

    def check_meta_sync(self, meta_key: str):
        """
        It checks if the local and server versions of a metadata file are the same

        :param meta_key: str
        :type meta_key: str
        :return: The return value is a list of the names of the files in the directory.
        """
        try:
            local_csv, server_csv = getattr(
                self, "local_" + meta_key + "_csv"
            ), getattr(self, "server_" + meta_key + "_csv")
            common_keys = np.intersect1d(local_csv.columns, server_csv.columns)
            assert local_csv[common_keys].equals(server_csv[common_keys])
            logging.info(f"Local and server versions of {meta_key} are synced.")
        except AssertionError:
            logging.error(f"Local and server versions of {meta_key} are not synced.")
            return

    def get_col_names(self, local_csv: str):
        """Return a dictionary with the project-specific column names of a csv of interest
        This function helps matching the schema format without modifying the column names of the original csv.

        :param project: The project object
        :param local_csv: a string of the name of the local csv of interest
        :return: a dictionary with the names of the columns
        """

        # Get project-specific server info
        project_name = self.project.Project_name

        if "sites" in local_csv:
            # Get spyfish specific column names
            if project_name == "Spyfish_Aotearoa":
                col_names_sites = get_spyfish_col_names("sites")

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
                col_names_movies = get_spyfish_col_names("movies")

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

    def check_movies_csv(
        self,
        review_method: str,
        gpu_available: bool = False,
    ):
        """
        > The function `check_movies_csv` loads the csv with movies information and checks if it is empty

        :param review_method: The method used to review the movies
        :param gpu_available: Boolean, whether or not a GPU is available
        """

        # Load the csv with movies information
        df = pd.read_csv(self.db_info["local_movies_csv"])

        # Get project-specific column names
        col_names = self.get_col_names("local_movies_csv")

        # Set project-specific column names of interest
        col_fps = col_names["fps"]
        col_duration = col_names["duration"]
        col_sampling_start = col_names["sampling_start"]
        col_sampling_end = col_names["sampling_end"]
        col_fpath = col_names["fpath"]

        if review_method.startswith("Basic"):
            # Check if fps or duration is missing from any movie
            if (
                not df[[col_fps, col_duration, col_sampling_start, col_sampling_end]]
                .isna()
                .any()
                .any()
            ):
                logging.info(
                    "There are no empty entries for fps, duration and sampling information"
                )

            else:
                # Create a df with only those rows with missing fps/duration
                df_missing = df[
                    df[col_fps].isna() | df[col_duration].isna()
                ].reset_index(drop=True)

                ##### Select only movies that can be mapped ####
                # Merge the missing fps/duration df with the available movies
                df_missing = df_missing.merge(
                    self.server_movies_csv[["filename", "exists", "spath"]],
                    on=["filename"],
                    how="left",
                )

                if df_missing.exists.isnull().values.any():
                    # Replace na with False
                    df_missing["exists"] = df_missing["exists"].fillna(False)

                    logging.info(
                        f"Only # {df_missing[df_missing['exists']].shape[0]} out of # {df_missing[~df_missing['exists']].shape[0]} movies with missing information are available. Proceeding to retrieve fps and duration info for only those {df_missing[df_missing['exists']].shape[0]} available movies."
                    )

                    # Select only available movies
                    df_missing = df_missing[df_missing["exists"]].reset_index(drop=True)

                # Rename column to match the movie_path format
                df_missing = df_missing.rename(
                    columns={
                        "spath": "movie_path",
                    }
                )

                logging.info("Getting the fps and duration of the movies")
                # Read the movies and overwrite the existing fps and duration info
                df_missing[[col_fps, col_duration]] = pd.DataFrame(
                    [
                        movie_utils.get_fps_duration(i)
                        for i in tqdm(
                            df_missing["movie_path"], total=df_missing.shape[0]
                        )
                    ],
                    columns=[col_fps, col_duration],
                )

                # Add the missing info to the original df based on movie ids
                df.set_index("movie_id", inplace=True)
                df_missing.set_index("movie_id", inplace=True)
                df.update(df_missing)
                df.reset_index(drop=False, inplace=True)

        else:
            logging.info("Retrieving the paths to access the movies")
            # Add a column with the path (or url) where the movies can be accessed from
            df["movie_path"] = pd.Series(
                [
                    movie_utils.get_movie_path(i, self.db_info, self.project)
                    for i in tqdm(df[col_fpath], total=df.shape[0])
                ]
            )

            logging.info("Getting the fps and duration of the movies")
            # Read the movies and overwrite the existing fps and duration info
            df[[col_fps, col_duration]] = pd.DataFrame(
                [
                    movie_utils.get_fps_duration(i)
                    for i in tqdm(df["movie_path"], total=df.shape[0])
                ],
                columns=[col_fps, col_duration],
            )

            logging.info("Standardising the format, frame rate and codec of the movies")

            # Convert movies to the right format, frame rate or codec and upload them to the project's server/storage
            [
                movie_utils.standarise_movie_format(
                    i, j, k, self.db_info, self.project, gpu_available
                )
                for i, j, k in tqdm(
                    zip(df["movie_path"], df["filename"], df[col_fpath]),
                    total=df.shape[0],
                )
            ]

            # Drop unnecessary columns
            df = df.drop(columns=["movie_path"])

        # Fill out missing sampling start information
        df.loc[df[col_sampling_start].isna(), col_sampling_start] = 0.0

        # Fill out missing sampling end information
        df.loc[df[col_sampling_end].isna(), col_sampling_end] = df[col_duration]

        # Prevent sampling end times longer than actual movies
        if (df[col_sampling_end] > df[col_duration]).any():
            mov_list = df[df[col_sampling_end] > df[col_duration]].filename.unique()
            raise ValueError(
                f"The sampling_end times of the following movies are longer than the actual movies {mov_list}"
            )

        # Save the updated df locally
        df.to_csv(self.db_info["local_movies_csv"], index=False)
        logging.info("The local movies.csv file has been updated")

        # Save the updated df in the server
        self.update_csv_server(
            orig_csv="server_movies_csv",
            updated_csv="local_movies_csv",
        )

    def check_movies_from_server(self):
        """
        It takes in a dataframe of movies and a dictionary of database information, and returns two
        dataframes: one of movies missing from the server, and one of movies missing from the csv

        :param db_info_dict: a dictionary with the following keys:
        :param project: the project object
        """
        # Load the csv with movies information
        movies_df = pd.read_csv(self.db_info["local_movies_csv"])

        # Check if the project is the Spyfish Aotearoa
        if self.project.Project_name == "Spyfish_Aotearoa":
            # Retrieve movies that are missing info in the movies.csv
            missing_info = check_spyfish_movies(movies_df, self.db_info)

        # Find out files missing from the Server
        missing_from_server = missing_info[missing_info["_merge"] == "left_only"]

        logging.info(f"There are {len(missing_from_server.index)} movies missing")

        # Find out files missing from the csv
        missing_from_csv = missing_info[
            missing_info["_merge"] == "right_only"
        ].reset_index(drop=True)

        logging.info(
            f"There are {len(missing_from_csv.index)} movies missing from movies.csv. Their filenames are:{missing_from_csv.filename.unique()}"
        )

        return missing_from_server, missing_from_csv

    def check_species_csv(self):
        """
        > The function `check_species_csv` loads the csv with species information and checks if it is empty

        :param db_info_dict: a dictionary with the following keys:
        :param project: The project name
        """
        # Load the csv with movies information
        species_df = pd.read_csv(self.db_info["local_species_csv"])

        # Retrieve the names of the basic columns in the sql db
        conn = db_utils.create_connection(self.db_info["db_path"])
        data = conn.execute("SELECT * FROM species")
        field_names = [i[0] for i in data.description]

        # Select the basic fields for the db check
        df_to_db = species_df[[c for c in species_df.columns if c in field_names]]

        # Roadblock to prevent empty lat/long/datum/countrycode
        db_utils.test_table(df_to_db, "species", df_to_db.columns)

        logging.info("The species dataframe is complete")

    def choose_footage(self, start_path: str = ".", folder_type: str = ""):
        if self.project.server == "AWS":
            available_movies_df = movie_utils.retrieve_movie_info_from_server(
                project=self.project, db_info_dict=self.db_info
            )
            movie_dict = {
                name: movie_utils.get_movie_path(f_path, self.db_info, self.project)
                for name, f_path in available_movies_df[["filename", "fpath"]].values
            }

            movie_widget = widgets.SelectMultiple(
                options=[(name, movie) for name, movie in movie_dict.items()],
                description="Select movie(s):",
                ensure_option=False,
                disabled=False,
                layout=widgets.Layout(width="50%"),
                style={"description_width": "initial"},
            )

            display(movie_widget)
            return movie_widget

        else:
            # Specify the output folder
            fc = FileChooser(start_path)
            fc.title = f"Choose location of {folder_type}"
            display(fc)
            return fc

    def check_sites_meta(self):
        # TODO: code for processing sites metadata (t1_utils.check_sites_csv)
        pass

    def update_new_deployments(
        self, deployment_selected: widgets.Widget, event_date: widgets.Widget
    ):
        """
        It takes a deployment, downloads all the movies from that deployment, concatenates them, uploads the
        concatenated video to the S3 bucket, and deletes the raw movies from the S3 bucket

        :param deployment_selected: the deployment you want to concatenate
        :param db_info_dict: a dictionary with the following keys:
        :param event_date: the date of the event you want to concatenate
        """
        for deployment_i in deployment_selected.value:
            logging.info(
                f"Starting to concatenate {deployment_i} out of {len(deployment_selected.value)} deployments selected"
            )

            # Get a dataframe of movies from the deployment
            movies_s3_pd = server_utils.get_matching_s3_keys(
                self.db_info["client"],
                self.db_info["bucket"],
                prefix=deployment_i,
                suffix=movie_utils.get_movie_extensions(),
            )

            # Create a list of the list of movies inside the deployment selected
            movie_files_server = movies_s3_pd.Key.unique().tolist()

            if len(movie_files_server) < 2:
                logging.info(
                    f"Deployment {deployment_i} will not be concatenated because it only has {movies_s3_pd.Key.unique()}"
                )
            else:
                # Concatenate the files if multiple
                logging.info(f"The files {movie_files_server} will be concatenated")

                # Start text file and list to keep track of the videos to concatenate
                textfile_name = "a_file.txt"
                textfile = open(textfile_name, "w")
                video_list = []

                for movie_i in sorted(movie_files_server):
                    # Specify the temporary output of the go pro file
                    movie_i_output = movie_i.split("/")[-1]

                    # Download the files from the S3 bucket
                    if not os.path.exists(movie_i_output):
                        server_utils.download_object_from_s3(
                            client=self.db_info["client"],
                            bucket=self.db_info["bucket"],
                            key=movie_i,
                            filename=movie_i_output,
                        )
                    # Keep track of the videos to concatenate
                    textfile.write("file '" + movie_i_output + "'" + "\n")
                    video_list.append(movie_i_output)
                textfile.close()

                # Save eventdate as str
                EventDate_str = event_date.value.strftime("%d_%m_%Y")

                # Specify the name of the concatenated video
                filename = deployment_i.split("/")[-1] + "_" + EventDate_str + ".MP4"

                # Concatenate the files
                if not os.path.exists(filename):
                    logging.info("Concatenating ", filename)

                    # Concatenate the videos
                    subprocess.call(
                        [
                            "ffmpeg",
                            "-f",
                            "concat",
                            "-safe",
                            "0",
                            "-i",
                            "a_file.txt",
                            "-c:a",
                            "copy",
                            "-c:v",
                            "h264",
                            "-crf",
                            "22",
                            filename,
                        ]
                    )

                # Upload the concatenated video to the S3
                server_utils.upload_file_to_s3(
                    self.db_info["client"],
                    bucket=self.db_info["bucket"],
                    key=deployment_i + "/" + filename,
                    filename=filename,
                )

                logging.info(f"{filename} successfully uploaded to {deployment_i}")

                # Delete the raw videos downloaded from the S3 bucket
                for f in video_list:
                    os.remove(f)

                # Delete the text file
                os.remove(textfile_name)

                # Delete the concat video
                os.remove(filename)

                # Delete the movies from the S3 bucket
                for movie_i in sorted(movie_files_server):
                    server_utils.delete_file_from_s3(
                        client=self.db_info["client"],
                        bucket=self.db_info["bucket"],
                        key=movie_i,
                    )

    # t2
    def upload_movies(self, movie_list: list):
        """
        It uploads the new movies to the SNIC server and creates new rows to be updated
        with movie metadata and saved into movies.csv

        :param db_info_dict: a dictionary with the following keys:
        :param movie_list: list of new movies that are to be added to movies.csv
        """
        # Get number of new movies to be added
        movie_folder = self.project.movie_folder
        number_of_movies = len(movie_list)
        # Get current movies
        movies_df = pd.read_csv(self.db_info["local_movies_csv"])
        # Set up a new row for each new movie
        new_movie_rows_sheet = ipysheet.sheet(
            rows=number_of_movies,
            columns=movies_df.shape[1],
            column_headers=movies_df.columns.tolist(),
        )
        if len(movie_list) == 0:
            logging.error("No valid movie found to upload.")
            return
        for index, movie in enumerate(movie_list):
            if self.project.server == "SNIC":
                # Specify volume allocated by SNIC
                snic_path = "/mimer/NOBACKUP/groups/snic2021-6-9"
                remote_fpath = Path(f"{snic_path}/tmp_dir/", movie[1])
            else:
                remote_fpath = Path(f"{movie_folder}", movie[1])
            if os.path.exists(remote_fpath):
                logging.info(
                    "Filename "
                    + str(movie[1])
                    + " already exists on SNIC, try again with a new file"
                )
                return
            else:
                # process video
                stem = "processed"
                p = Path(movie[0])
                processed_video_path = p.with_name(f"{p.stem}_{stem}{p.suffix}").name
                logging.info("Movie to be uploaded: " + processed_video_path)
                ffmpeg.input(p).output(
                    processed_video_path,
                    crf=22,
                    pix_fmt="yuv420p",
                    vcodec="libx264",
                    threads=4,
                ).run(capture_stdout=True, capture_stderr=True, overwrite_output=True)

                if self.project.server == "SNIC":
                    server_utils.upload_object_to_snic(
                        self.db_info["sftp_client"],
                        str(processed_video_path),
                        str(remote_fpath),
                    )
                elif self.project.server in ["LOCAL", "TEMPLATE"]:
                    shutil.copy2(str(processed_video_path), str(remote_fpath))
                logging.info("movie uploaded\n")
            # Fetch movie metadata that can be calculated from movie file
            fps, duration = movie_utils.get_fps_duration(movie[0])
            movie_id = str(max(movies_df["movie_id"]) + 1)
            ipysheet.cell(index, 0, movie_id)
            ipysheet.cell(index, 1, movie[1])
            ipysheet.cell(index, 2, "-")
            ipysheet.cell(index, 3, "-")
            ipysheet.cell(index, 4, "-")
            ipysheet.cell(index, 5, fps)
            ipysheet.cell(index, 6, duration)
            ipysheet.cell(index, 7, "-")
            ipysheet.cell(index, 8, "-")
        logging.info("All movies uploaded:\n")
        logging.info(
            "Complete this sheet by filling the missing info on the movie you just uploaded"
        )
        display(new_movie_rows_sheet)
        return new_movie_rows_sheet

    def add_movies(self):
        """
        > It creates a button that, when clicked, creates a new button that, when clicked, saves the
        changes to the local csv file of the new movies that should be added. It creates a metadata row
        for each new movie, which should be filled in by the user before uploading can continue.
        """
        movie_list = t_utils.choose_new_videos_to_upload()
        button = widgets.Button(
            description="Click to upload movies",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
            style={"width": "initial"},
        )

        def on_button_clicked(b):
            new_sheet = self.upload_movies(movie_list)
            button2 = widgets.Button(
                description="Save changes",
                disabled=False,
                display="flex",
                flex_flow="column",
                align_items="stretch",
                style={"width": "initial"},
            )

            def on_button_clicked2(b):
                movies_df = pd.read_csv(self.db_info["local_movies_csv"])
                new_movie_rows_df = ipysheet.to_dataframe(new_sheet)
                self.local_movies_csv = pd.concat(
                    [movies_df, new_movie_rows_df], ignore_index=True
                )
                logging.info("Changed saved locally")

            button2.on_click(on_button_clicked2)
            display(button2)

        button.on_click(on_button_clicked)
        # t2_utils.upload_new_movies_to_snic
        # t2_utils.update_csv
        # t2_utils.sync_server_csv
        display(button)

    def select_survey(self):
        """
        This function allows the user to select an existing survey from a dropdown menu or create a new
        survey by filling out a series of widgets

        :param db_info_dict: a dictionary with the following keys:
        :type db_info_dict: dict
        :return: A widget with a dropdown menu with two options: 'Existing' and 'New survey'.
        """
        # Load the csv with surveys information
        surveys_df = pd.read_csv(self.db_info["local_surveys_csv"])

        # Existing Surveys
        exisiting_surveys = surveys_df.SurveyName.unique()

        def f(Existing_or_new):
            if Existing_or_new == "Existing":
                survey_widget = widgets.Dropdown(
                    options=exisiting_surveys,
                    description="Survey Name:",
                    disabled=False,
                    layout=widgets.Layout(width="80%"),
                    style={"description_width": "initial"},
                )

                display(survey_widget)

                return survey_widget

            if Existing_or_new == "New survey":
                # Load the csv with with sites and survey choices
                choices_df = pd.read_csv(self.db_info["local_choices_csv"])

                # Save the new survey responses into a dict
                survey_info = {
                    # Write the name of the encoder
                    "EncoderName": t_utils.record_encoder(),
                    # Select the start date of the survey
                    "SurveyStartDate": t_utils.select_SurveyStartDate(),
                    # Write the name of the survey
                    "SurveyName": t_utils.write_SurveyName(),
                    # Select the DOC office
                    "OfficeName": t_utils.select_OfficeName(
                        choices_df.OfficeName.dropna().unique().tolist()
                    ),
                    # Write the name of the contractor
                    "ContractorName": t_utils.write_ContractorName(),
                    # Write the number of the contractor
                    "ContractNumber": t_utils.write_ContractNumber(),
                    # Write the link to the contract
                    "LinkToContract": t_utils.write_LinkToContract(),
                    # Record the name of the survey leader
                    "SurveyLeaderName": t_utils.write_SurveyLeaderName(),
                    # Select the name of the linked Marine Reserve
                    "LinkToMarineReserve": t_utils.select_LinkToMarineReserve(
                        choices_df.MarineReserve.dropna().unique().tolist()
                    ),
                    # Specify if survey is single species
                    "FishMultiSpecies": t_utils.select_FishMultiSpecies(),
                    # Specify how the survey was stratified
                    "StratifiedBy": t_utils.select_StratifiedBy(
                        choices_df.StratifiedBy.dropna().unique().tolist()
                    ),
                    # Select if survey is part of long term monitoring
                    "IsLongTermMonitoring": t_utils.select_IsLongTermMonitoring(),
                    # Specify the site selection of the survey
                    "SiteSelectionDesign": t_utils.select_SiteSelectionDesign(
                        choices_df.SiteSelection.dropna().unique().tolist()
                    ),
                    # Specify the unit selection of the survey
                    "UnitSelectionDesign": t_utils.select_UnitSelectionDesign(
                        choices_df.UnitSelection.dropna().unique().tolist()
                    ),
                    # Select the type of right holder of the survey
                    "RightsHolder": t_utils.select_RightsHolder(
                        choices_df.RightsHolder.dropna().unique().tolist()
                    ),
                    # Write who can access the videos/resources
                    "AccessRights": t_utils.select_AccessRights(),
                    # Write a description of the survey design and objectives
                    "SurveyVerbatim": t_utils.write_SurveyVerbatim(),
                    # Select the type of BUV
                    "BUVType": t_utils.select_BUVType(
                        choices_df.BUVType.dropna().unique().tolist()
                    ),
                    # Write the link to the pictures
                    "LinkToPicture": t_utils.write_LinkToPicture(),
                    # Write the name of the vessel
                    "Vessel": t_utils.write_Vessel(),
                    # Write the link to the fieldsheets
                    "LinkToFieldSheets": t_utils.write_LinkToFieldSheets(),
                    # Write the link to LinkReport01
                    "LinkReport01": t_utils.write_LinkReport01(),
                    # Write the link to LinkReport02
                    "LinkReport02": t_utils.write_LinkReport02(),
                    # Write the link to LinkReport03
                    "LinkReport03": t_utils.write_LinkReport03(),
                    # Write the link to LinkReport04
                    "LinkReport04": t_utils.write_LinkReport04(),
                    # Write the link to LinkToOriginalData
                    "LinkToOriginalData": t_utils.write_LinkToOriginalData(),
                }

                return survey_info

        w = widgets.interactive(
            f,
            Existing_or_new=widgets.Dropdown(
                options=["Existing", "New survey"],
                description="Existing or new survey:",
                disabled=False,
                layout=widgets.Layout(width="90%"),
                style={"description_width": "initial"},
            ),
        )

        display(w)
        return w

    def confirm_survey(self, survey_i):
        """
        It takes the survey information and checks if it's a new survey or an existing one. If it's a new
        survey, it saves the information in the survey csv file. If it's an existing survey, it prints the
        information for the pre-existing survey

        :param survey_i: the survey widget
        :param db_info_dict: a dictionary with the following keys:
        """

        correct_button = widgets.Button(
            description="Yes, details are correct",
            layout=widgets.Layout(width="25%"),
            style={"description_width": "initial"},
            button_style="danger",
        )

        wrong_button = widgets.Button(
            description="No, I will go back and fix them",
            layout=widgets.Layout(width="45%"),
            style={"description_width": "initial"},
            button_style="danger",
        )

        # If new survey, review details and save changes in survey csv server
        if isinstance(survey_i.result, dict):
            # Save the responses as a new row for the survey csv file
            new_survey_row_dict = {
                key: (
                    value.value
                    if hasattr(value, "value")
                    else value.result
                    if isinstance(value.result, int)
                    else value.result.value
                )
                for key, value in survey_i.result.items()
            }
            new_survey_row = pd.DataFrame.from_records(new_survey_row_dict, index=[0])

            # Load the csv with sites and survey choices
            choices_df = pd.read_csv(self.db_info["local_choices_csv"])

            # Get prepopulated fields for the survey
            new_survey_row["OfficeContact"] = choices_df[
                choices_df["OfficeName"] == new_survey_row.OfficeName.values[0]
            ]["OfficeContact"].values[0]
            new_survey_row[["SurveyLocation", "Region"]] = choices_df[
                choices_df["MarineReserve"]
                == new_survey_row.LinkToMarineReserve.values[0]
            ][["MarineReserveAbreviation", "Region"]].values[0]
            new_survey_row["DateEntry"] = datetime.date.today()
            new_survey_row["SurveyType"] = "BUV"
            new_survey_row["SurveyID"] = (
                new_survey_row["SurveyLocation"]
                + "_"
                + new_survey_row["SurveyStartDate"].values[0].strftime("%Y%m%d")
                + "_"
                + new_survey_row["SurveyType"]
            )

            # Review details
            print("The details of the new survey are:")
            for ind in new_survey_row.T.index:
                print(ind, "-->", new_survey_row.T[0][ind])

            # Save changes in survey csv locally and in the server
            async def f(new_survey_row):
                x = await t_utils.wait_for_change(
                    correct_button, wrong_button
                )  # <---- Pass both buttons into the function
                if (
                    x == "Yes, details are correct"
                ):  # <--- use if statement to trigger different events for the two buttons
                    # Load the csv with sites information
                    surveys_df = pd.read_csv(self.db_info["local_surveys_csv"])

                    # Drop unnecessary columns
                    #                 new_survey_row = new_survey_row.drop(columns=['ShortFolder'])

                    # Check the columns are the same
                    diff_columns = list(
                        set(surveys_df.columns.sort_values().values)
                        - set(new_survey_row.columns.sort_values().values)
                    )

                    if len(diff_columns) > 0:
                        logging.error(
                            f"The {diff_columns} columns are missing from the survey information."
                        )
                        raise

                    # Check if the survey exist in the csv
                    if (
                        new_survey_row.SurveyID.unique()[0]
                        in surveys_df.SurveyID.unique()
                    ):
                        logging.error(
                            f"The survey {new_survey_row.SurveyID.unique()[0]} already exists in the database."
                        )
                        raise

                    print("Updating the new survey information.")

                    # Add the new row to the choices df
                    surveys_df = surveys_df.append(new_survey_row, ignore_index=True)

                    # Save the updated df locally
                    surveys_df.to_csv(self.db_info["local_surveys_csv"], index=False)

                    # Save the updated df in the server
                    server_utils.upload_file_to_s3(
                        self.db_info["client"],
                        bucket=self.db_info["bucket"],
                        key=self.db_info["server_surveys_csv"],
                        filename=self.db_info["local_surveys_csv"].__str__(),
                    )

                    print("Survey information updated!")

                else:
                    print("Come back when the data is tidy!")

        # If existing survey print the info for the pre-existing survey
        else:
            # Load the csv with surveys information
            surveys_df = pd.read_csv(self.db_info["local_surveys_csv"])

            # Select the specific survey info
            new_survey_row = surveys_df[
                surveys_df["SurveyName"] == survey_i.result.value
            ].reset_index(drop=True)

            print("The details of the selected survey are:")
            for ind in new_survey_row.T.index:
                print(ind, "-->", new_survey_row.T[0][ind])

            async def f(new_survey_row):
                x = await t_utils.wait_for_change(
                    correct_button, wrong_button
                )  # <---- Pass both buttons into the function
                if (
                    x == "Yes, details are correct"
                ):  # <--- use if statement to trigger different events for the two buttons
                    print("Great, you can start uploading the movies.")

                else:
                    print("Come back when the data is tidy!")

        print("")
        print("")
        print("Are the survey details above correct?")
        display(
            widgets.HBox([correct_button, wrong_button])
        )  # <----Display both buttons in an HBox
        asyncio.create_task(f(new_survey_row))

    def add_sites(self):
        pass

    def add_species(self):
        pass

    def view_annotations(self, folder_path: str, annotation_classes: list):
        """
        > This function takes in a folder path and a list of annotation classes and returns a widget that
        allows you to view the annotations in the folder

        :param folder_path: The path to the folder containing the images you want to annotate
        :type folder_path: str
        :param annotation_classes: list of strings
        :type annotation_classes: list
        :return: A list of dictionaries, each dictionary containing the following keys
                 - 'image_path': the path to the image
                 - 'annotations': a list of dictionaries, each dictionary containing the following keys:
                 - 'class': the class of the annotation
                 - 'bbox': the bounding box of the annotation
        """
        return t_utils.get_annotations_viewer(
            folder_path, species_list=annotation_classes
        )

    def select_deployment(self, survey_i):
        """
        This function allows the user to select a deployment from a survey of interest

        :param project: the name of the project
        :param db_info_dict: a dictionary with the following keys:
        :type db_info_dict: dict
        :param survey_i: the index of the survey you want to download
        """
        # Load the csv with with sites and survey choices
        choices_df = pd.read_csv(self.db_info["local_choices_csv"])

        # Read surveys csv
        surveys_df = pd.read_csv(
            self.db_info["local_surveys_csv"], parse_dates=["SurveyStartDate"]
        )

        # Get the name of the survey
        survey_name = t_utils.get_survey_name(survey_i)

        # Save the SurveyID that match the survey name
        survey_row = surveys_df[surveys_df["SurveyName"] == survey_name].reset_index(
            drop=True
        )

        # Save the year of the survey
        #     survey_year = survey_row["SurveyStartDate"].values[0].strftime("%Y")
        survey_year = survey_row["SurveyStartDate"].dt.year.values[0]

        # Get prepopulated fields for the survey
        survey_row[["ShortFolder"]] = choices_df[
            choices_df["MarineReserve"] == survey_row.LinkToMarineReserve.values[0]
        ][["ShortFolder"]].values[0]

        # Save the "server filename" of the survey
        short_marine_reserve = survey_row["ShortFolder"].values[0]

        # Save the "server filename" of the survey
        survey_server_name = short_marine_reserve + "-buv-" + str(survey_year) + "/"

        # Retrieve deployments info from the survey of interest
        deployments_server_name = self.db_info["client"].list_objects(
            Bucket=self.db_info["bucket"], Prefix=survey_server_name, Delimiter="/"
        )
        # Convert info to dataframe
        deployments_server_name = pd.DataFrame.from_dict(
            deployments_server_name["CommonPrefixes"]
        )

        # Select only the name of the "deployment folder"
        deployments_server_name_list = deployments_server_name.Prefix.str.split(
            "/"
        ).str[1]

        # Widget to select the deployment of interest
        deployment_widget = widgets.SelectMultiple(
            options=deployments_server_name_list,
            description="New deployment:",
            disabled=False,
            layout=widgets.Layout(width="50%"),
            style={"description_width": "initial"},
        )
        display(deployment_widget)
        return deployment_widget, survey_row, survey_server_name

    def check_deployment(
        self,
        deployment_selected: widgets.Widget,
        deployment_date: widgets.Widget,
        survey_server_name: str,
        survey_i,
    ):
        """
        This function checks if the deployment selected by the user is already in the database. If it is, it
        will raise an error. If it is not, it will return the deployment filenames

        :param deployment_selected: a list of the deployment names selected by the user
        :param deployment_date: the date of the deployment
        :param survey_server_name: The name of the survey in the server
        :param db_info_dict: a dictionary containing the following keys:
        :param survey_i: the index of the survey you want to upload to
        :return: A list of deployment filenames
        """
        # Ensure at least one deployment has been selected
        if not deployment_selected.value:
            logging.error("Please select a deployment.")
            raise

        # Get list of movies available in server from that survey
        deployments_in_server_df = server_utils.get_matching_s3_keys(
            self.db_info["client"], self.db_info["bucket"], prefix=survey_server_name
        )

        # Get a list of the movies in the server
        files_in_server = deployments_in_server_df.Key.str.split("/").str[-1].to_list()
        deployments_in_server = [
            file
            for file in files_in_server
            if file[-3:] in movie_utils.get_movie_extensions()
        ]

        # Read surveys csv
        surveys_df = pd.read_csv(
            self.db_info["local_surveys_csv"], parse_dates=["SurveyStartDate"]
        )

        # Get the name of the survey
        survey_name = t_utils.get_survey_name(survey_i)

        # Save the SurveyID that match the survey name
        SurveyID = surveys_df[
            surveys_df["SurveyName"] == survey_name
        ].SurveyID.to_list()[0]

        # Read movie.csv info
        movies_df = pd.read_csv(self.db_info["local_movies_csv"])

        # Get a list of deployment names from the csv of the survey of interest
        deployment_in_csv = movies_df[
            movies_df["SurveyID"] == SurveyID
        ].filename.to_list()

        # Save eventdate as str
        EventDate_str = deployment_date.value.strftime("%d_%m_%Y")

        deployment_filenames = []
        for deployment_i in deployment_selected.value:
            # Specify the name of the deployment
            deployment_name = deployment_i + "_" + EventDate_str

            if deployment_name in deployment_in_csv:
                logging.error(
                    f"Deployment {deployment_name} already exist in the csv file reselect new deployments before going any further!"
                )
                raise

            # Specify the filename of the deployment
            filename = deployment_name + ".MP4"

            if filename in deployments_in_server:
                logging.error(
                    f"Deployment {deployment_name} already exist in the server reselect new deployments before going any further!"
                )
                raise

            else:
                deployment_filenames = deployment_filenames + [filename]
        print(
            "There is no existing information in the database for",
            deployment_filenames,
            "You can continue uploading the information.",
        )

        return deployment_filenames

    def update_new_deployments(
        self,
        deployment_filenames: list,
        survey_server_name: str,
        deployment_date: widgets.Widget,
    ):
        """
        It takes a list of filenames, a dictionary with the database information, the name of the server,
        and the date of the deployment, and it returns a list of the movies in the server

        :param deployment_filenames: a list of the filenames of the videos you want to concatenate
        :param db_info_dict: a dictionary with the following keys:
        :param survey_server_name: the name of the folder in the server where the survey is stored
        :param deployment_date: the date of the deployment
        :return: A list of the movies in the server
        """
        for deployment_i in deployment_filenames:
            # Save eventdate as str
            EventDate_str = deployment_date.value.strftime("%d_%m_%Y")

            # Get the site information from the filename
            site_deployment = deployment_i.replace("_" + EventDate_str + ".MP4", "")
            print(site_deployment)

            # Specify the folder with files in the server
            deployment_folder = survey_server_name + site_deployment

            # Retrieve files info from the deployment folder of interest
            deployments_files = server_utils.get_matching_s3_keys(
                self.db_info["client"], self.db_info["bucket"], prefix=deployment_folder
            )
            # Get a list of the movies in the server
            files_server = deployments_files.Key.to_list()
            movie_files_server = [
                file
                for file in files_server
                if file[-3:] in movie_utils.get_movie_extensions()
            ]

            # Concatenate the files if multiple
            if len(movie_files_server) > 1:
                print("The files", movie_files_server, "will be concatenated")

                # Save filepaths in a text file
                textfile = open("a_file.txt", "w")

                for movie_i in sorted(movie_files_server):
                    temp_i = movie_i.split("/")[2]
                    server_utils.download_object_from_s3(
                        client=self.db_info["client"],
                        bucket=self.db_info["bucket"],
                        key=movie_i,
                        filename=temp_i,
                    )

                    textfile.write("file '" + temp_i + "'" + "\n")
                textfile.close()

                if not os.path.exists(deployment_i):
                    print("Concatenating ", deployment_i)

                    # Concatenate the videos
                    subprocess.call(
                        [
                            "ffmpeg",
                            "-f",
                            "concat",
                            "-safe",
                            "0",
                            "-i",
                            "a_file.txt",
                            "-c",
                            "copy",
                            deployment_i,
                        ]
                    )

                    print(deployment_i, "concatenated successfully")

            # Change the filename if only one file
            else:
                print("WIP")

            return movie_files_server

    def record_deployment_info(self, deployment_filenames: list):
        """
        This function takes in a list of deployment filenames and a dictionary of database information, and
        returns a dictionary of deployment information.

        :param deployment_filenames: a list of the filenames of the movies you want to add to the database
        :param db_info_dict: a dictionary with the following keys:
        :return: A dictionary with the deployment info
        """

        for deployment_i in deployment_filenames:
            # Estimate the fps and length info
            fps, duration = movie_utils.get_length(deployment_i, os.getcwd())

            # Read csv as pd
            movies_df = pd.read_csv(self.db_info["local_movies_csv"])

            # Load the csv with with sites and survey choices
            choices_df = pd.read_csv(self.db_info["local_choices_csv"])

            deployment_info = {
                # Select the start of the sampling
                "SamplingStart": t_utils.select_SamplingStart(duration),
                # Select the end of the sampling
                "SamplingEnd": t_utils.select_SamplingEnd(duration),
                # Specify if deployment is bad
                "IsBadDeployment": t_utils.select_IsBadDeployment(),
                # Write the number of the replicate within the site
                "ReplicateWithinSite": t_utils.write_ReplicateWithinSite(),
                # Select the person who recorded this deployment
                "RecordedBy": t_utils.select_RecordedBy(movies_df.RecordedBy.unique()),
                # Select depth stratum of the deployment
                "DepthStrata": t_utils.select_DepthStrata(),
                # Select the depth of the deployment
                "Depth": t_utils.select_Depth(),
                # Select the underwater visibility
                "UnderwaterVisibility": t_utils.select_UnderwaterVisibility(
                    choices_df.UnderwaterVisibility.dropna().unique().tolist()
                ),
                # Select the time in
                "TimeIn": t_utils.deployment_TimeIn(),
                # Select the time out
                "TimeOut": t_utils.deployment_TimeOut(),
                # Add any comment related to the deployment
                "NotesDeployment": t_utils.write_NotesDeployment(),
                # Select the theoretical duration of the deployment
                "DeploymentDurationMinutes": t_utils.select_DeploymentDurationMinutes(),
                # Write the type of habitat
                "Habitat": t_utils.write_Habitat(),
                # Write the number of NZMHCS_Abiotic
                "NZMHCS_Abiotic": t_utils.write_NZMHCS_Abiotic(),
                # Write the number of NZMHCS_Biotic
                "NZMHCS_Biotic": t_utils.write_NZMHCS_Biotic(),
                # Select the level of the tide
                "TideLevel": t_utils.select_TideLevel(
                    choices_df.TideLevel.dropna().unique().tolist()
                ),
                # Describe the weather of the deployment
                "Weather": t_utils.write_Weather(),
                # Select the model of the camera used
                "CameraModel": t_utils.select_CameraModel(
                    choices_df.CameraModel.dropna().unique().tolist()
                ),
                # Write the camera lens and settings used
                "LensModel": t_utils.write_LensModel(),
                # Specify the type of bait used
                "BaitSpecies": t_utils.write_BaitSpecies(),
                # Specify the amount of bait used
                "BaitAmount": t_utils.select_BaitAmount(),
            }

            return deployment_info

    def confirm_deployment_details(
        self,
        deployment_names: list,
        survey_server_name: str,
        survey_i,
        deployment_info: list,
        movie_files_server: str,
        deployment_date: widgets.Widget,
    ):
        """
        This function takes the deployment information and the survey information and returns a dataframe
        with the deployment information

        :param deployment_names: list of deployment names
        :param survey_server_name: The name of the folder in the server where the survey is located
        :param db_info_dict: a dictionary with the following keys:
        :param survey_i: the survey name
        :param deployment_info: a list of dictionaries with the deployment information
        :param movie_files_server: the path to the folder with the movie files in the server
        :param deployment_date: The date of the deployment
        :return: The new_movie_row is a dataframe with the information of the new deployment.
        """

        for deployment_i in deployment_names:
            # Save the deployment responses as a new row for the movies csv file
            new_movie_row_dict = {
                key: (
                    value.value
                    if hasattr(value, "value")
                    else value.result
                    if isinstance(value.result, int)
                    else value.result.value
                )
                for key, value in deployment_info[
                    deployment_names.index(deployment_i)
                ].items()
            }

            new_movie_row = pd.DataFrame.from_records(new_movie_row_dict, index=[0])

            # Read movies csv
            movies_df = pd.read_csv(self.db_info["local_movies_csv"])

            # Get prepopulated fields for the movie deployment
            # Add movie id
            new_movie_row["movie_id"] = 1 + movies_df.movie_id.iloc[-1]

            # Read surveys csv
            surveys_df = pd.read_csv(self.db_info["local_surveys_csv"])

            # Save the name of the survey
            if isinstance(survey_i.result, dict):
                # Load the ncsv with survey information
                surveys_df = pd.read_csv(self.db_info["local_surveys_csv"])

                # Save the SurveyID of the last survey added
                new_movie_row["SurveyID"] = surveys_df.tail(1)["SurveyID"].values[0]

                # Save the name of the survey
                survey_name = surveys_df.tail(1)["SurveyName"].values[0]

            else:
                # Return the name of the survey
                survey_name = survey_i.result.value

                # Save the SurveyID that match the survey name
                new_movie_row["SurveyID"] = surveys_df[
                    surveys_df["SurveyName"] == survey_name
                ]["SurveyID"].values[0]

            # Get the site information from the filename
            site_deployment = "_".join(deployment_i.split("_")[:2])

            # Specify the folder with files in the server
            deployment_folder = survey_server_name + site_deployment

            # Create temporary prefix (s3 path) for concatenated video
            new_movie_row["prefix_conc"] = deployment_folder + "/" + deployment_i

            # Select previously processed movies within the same survey
            survey_movies_df = movies_df[
                movies_df["SurveyID"] == new_movie_row["SurveyID"][0]
            ].reset_index()

            # Create unit id
            if survey_movies_df.empty:
                # Start unit_id in 0
                new_movie_row["UnitID"] = surveys_df["SurveyID"].values[0] + "_0000"

            else:
                # Get the last unitID
                last_unitID = str(
                    survey_movies_df.sort_values("UnitID").tail(1)["UnitID"].values[0]
                )[-4:]

                # Add one more to the last UnitID
                next_unitID = str(int(last_unitID) + 1).zfill(4)

                # Add one more to the last UnitID
                new_movie_row["UnitID"] = (
                    surveys_df["SurveyID"].values[0] + "_" + next_unitID
                )

            # Estimate the fps and length info
            new_movie_row[["fps", "duration"]] = movie_utils.get_length(
                deployment_i, os.getcwd()
            )

            # Specify location of the concat video
            new_movie_row["concat_video"] = deployment_i

            # Specify filename of the video
            new_movie_row["filename"] = deployment_i

            # Specify the site of the deployment
            new_movie_row["SiteID"] = site_deployment

            # Specify the date of the deployment
            new_movie_row["EventDate"] = deployment_date.value

            # Save the list of goprofiles
            new_movie_row["go_pro_files"] = [movie_files_server]

            # Extract year, month and day
            new_movie_row["Year"] = new_movie_row["EventDate"][0].year
            new_movie_row["Month"] = new_movie_row["EventDate"][0].month
            new_movie_row["Day"] = new_movie_row["EventDate"][0].day

            print("The details of the new deployment are:")
            for ind in new_movie_row.T.index:
                print(ind, "-->", new_movie_row.T[0][ind])

            return new_movie_row

    def upload_concat_movie(self, new_deployment_row: pd.DataFrame):
        """
        It uploads the concatenated video to the server and updates the movies csv file with the new
        information

        :param db_info_dict: a dictionary with the following keys:
        :param new_deployment_row: new deployment dataframe with the information of the new deployment
        """

        # Save to new deployment row df
        new_deployment_row["LinkToVideoFile"] = (
            "http://marine-buv.s3.ap-southeast-2.amazonaws.com/"
            + new_deployment_row["prefix_conc"][0]
        )

        # Remove temporary prefix for concatenated video and local path to concat_video
        new_movie_row = new_deployment_row.drop(["prefix_conc", "concat_video"], axis=1)

        # Load the csv with movies information
        movies_df = pd.read_csv(self.db_info["local_movies_csv"])

        # Check the columns are the same
        diff_columns = list(
            set(movies_df.columns.sort_values().values)
            - set(new_movie_row.columns.sort_values().values)
        )

        if len(diff_columns) > 0:
            logging.error(
                f"The {diff_columns} columns are missing from the information for the new deployment."
            )
            raise

        else:
            print(
                "Uploading the concatenated movie to the server.",
                new_deployment_row["prefix_conc"][0],
            )

            # Upload movie to the s3 bucket
            server_utils.upload_file_to_s3(
                client=self.db_info["client"],
                bucket=self.db_info["bucket"],
                key=new_deployment_row["prefix_conc"][0],
                filename=new_deployment_row["concat_video"][0],
            )

            print("Movie uploaded to", new_deployment_row["LinkToVideoFile"])

            # Add the new row to the movies df
            movies_df = movies_df.append(new_movie_row, ignore_index=True)

            # Save the updated df locally
            movies_df.to_csv(self.db_info["local_movies_csv"], index=False)

            # Save the updated df in the server
            server_utils.upload_file_to_s3(
                self.db_info["client"],
                bucket=self.db_info["bucket"],
                key=self.db_info["server_movies_csv"],
                filename=str(self.db_info["local_movies_csv"]),
            )

            # Remove temporary movie
            print("Movies csv file succesfully updated in the server.")

    # t3 / t4
    def select_random_clips(self, movie_i: str):
        """
        > The function `select_random_clips` takes in a movie name and a dictionary containing information
        about the database, and returns a dictionary containing the starting points of the clips and the
        length of the clips.

        :param movie_i: the name of the movie of interest
        :type movie_i: str
        :param db_info_dict: a dictionary containing the path to the database and the name of the database
        :type db_info_dict: dict
        :return: A dictionary with the starting points of the clips and the length of the clips.
        """
        # Create connection to db
        conn = db_utils.create_connection(self.db_info["db_path"])

        # Query info about the movie of interest
        movie_df = pd.read_sql_query(
            f"SELECT filename, duration, sampling_start, sampling_end FROM movies WHERE filename='{movie_i}'",
            conn,
        )

        # Select n number of clips at random
        def n_random_clips(clip_length, n_clips):
            # Create a list of starting points for n number of clips
            duration_movie = math.floor(movie_df["duration"].values[0])
            starting_clips = random.sample(
                range(0, duration_movie, clip_length), n_clips
            )

            # Seave the outputs in a dictionary
            random_clips_info = {
                # The starting points of the clips
                "clip_start_time": starting_clips,
                # The length of the clips
                "random_clip_length": clip_length,
            }

            logging.info(
                f"The initial seconds of the examples will be: {random_clips_info['clip_start_time']}"
            )

            return random_clips_info

        # Select the number of clips to upload
        clip_length_number = widgets.interactive(
            n_random_clips,
            clip_length=t_utils.select_clip_length(),
            n_clips=widgets.IntSlider(
                value=3,
                min=1,
                max=5,
                step=1,
                description="Number of random clips:",
                disabled=False,
                layout=widgets.Layout(width="40%"),
                style={"description_width": "initial"},
            ),
        )

        display(clip_length_number)
        return clip_length_number

    def create_example_clips(
        self,
        movie_i: str,
        movie_path: str,
        clip_selection,
        pool_size=4,
    ):
        """
        This function takes a movie and extracts clips from it, based on the start time and length of the
        clips

        :param movie_i: str, the name of the movie
        :type movie_i: str
        :param movie_path: the path to the movie
        :type movie_path: str
        :param db_info_dict: a dictionary containing the information of the database
        :type db_info_dict: dict
        :param project: the project object
        :param clip_selection: a dictionary with the following keys:
        :param pool_size: The number of parallel processes to run, defaults to 4 (optional)
        :return: The path of the clips
        """

        # Specify the starting seconds and length of the example clips
        clips_start_time = clip_selection.result["clip_start_time"]
        clip_length = clip_selection.result["random_clip_length"]

        # Get project-specific server info
        server = self.project.server

        # Specify the temp folder to host the clips
        output_clip_folder = movie_i + "_clips"
        if server == "SNIC":
            # Specify volume allocated by SNIC
            snic_path = "/mimer/NOBACKUP/groups/snic2021-6-9/"
            clips_folder = Path(snic_path, "tmp_dir", output_clip_folder)
        else:
            clips_folder = output_clip_folder

        # Create the folder to store the videos if not exist
        if not os.path.exists(clips_folder):
            Path(clips_folder).mkdir(parents=True, exist_ok=True)
            # Recursively add permissions to folders created
            [os.chmod(root, 0o777) for root, dirs, files in os.walk(clips_folder)]

        # Specify the number of parallel items
        pool = multiprocessing.Pool(pool_size)

        # Create empty list to keep track of new clips
        example_clips = []

        # Create the information for each clip and extract it (printing a progress bar)
        for start_time_i in clips_start_time:
            # Create the filename and path of the clip
            output_clip_name = (
                movie_i + "_clip_" + str(start_time_i) + "_" + str(clip_length) + ".mp4"
            )
            output_clip_path = Path(clips_folder, output_clip_name)

            # Add the path of the clip to the list
            example_clips = example_clips + [output_clip_path]

            # Extract the clips and store them in the folder
            pool.apply_async(
                t_utils.extract_example_clips,
                (
                    output_clip_path,
                    start_time_i,
                    clip_length,
                    movie_path,
                ),
            )

        pool.close()
        pool.join()

        logging.info("Clips extracted successfully")
        return example_clips

    def select_clip_n_len(self, movie_i: str):
        """
        This function allows the user to select the length of the clips to upload to the database

        :param movie_i: the name of the movie you want to upload
        :param db_info_dict: a dictionary containing the path to the database and the name of the database
        :return: The number of clips to upload
        """

        # Create connection to db
        conn = db_utils.create_connection(self.db_info["db_path"])

        # Query info about the movie of interest
        movie_df = pd.read_sql_query(
            f"SELECT filename, duration, sampling_start, sampling_end FROM movies WHERE filename='{movie_i}'",
            conn,
        )

        # Display in hours, minutes and seconds
        def to_clips(clip_length, clips_range):
            # Calculate the number of clips
            clips = int((clips_range[1] - clips_range[0]) / clip_length)

            logging.info(f"Number of clips to upload: {clips}")

            return clips

        # Select the number of clips to upload
        clip_length_number = widgets.interactive(
            to_clips,
            clip_length=t_utils.select_clip_length(),
            clips_range=widgets.IntRangeSlider(
                value=[movie_df.sampling_start.values, movie_df.sampling_end.values],
                min=0,
                max=int(movie_df.duration.values),
                step=1,
                description="Range in seconds:",
                style={"description_width": "initial"},
                layout=widgets.Layout(width="90%"),
            ),
        )

        display(clip_length_number)
        return clip_length_number

    def create_clips(
        self,
        available_movies_df: pd.DataFrame,
        movie_i: str,
        movie_path: str,
        clip_selection,
        modification_details: dict,
        gpu_available: bool,
        pool_size: int = 4,
    ):
        """
        This function takes a movie and extracts clips from it

        :param available_movies_df: the dataframe with the movies that are available for the project
        :param movie_i: the name of the movie you want to extract clips from
        :param movie_path: the path to the movie you want to extract clips from
        :param db_info_dict: a dictionary with the database information
        :param clip_selection: a ClipSelection object
        :param project: the project object
        :param modification_details: a dictionary with the following keys:
        :param gpu_available: True or False, depending on whether you have a GPU available to use
        :param pool_size: the number of threads to use to extract the clips, defaults to 4 (optional)
        :return: A dataframe with the clip_path, clip_filename, clip_length, upl_seconds, and clip_modification_details
        """

        # Filter the df for the movie of interest
        movie_i_df = available_movies_df[
            available_movies_df["filename"] == movie_i
        ].reset_index(drop=True)

        # Calculate the max number of clips available
        clip_length = clip_selection.kwargs["clip_length"]
        clip_numbers = clip_selection.result
        if "clips_range" in clip_selection.kwargs:
            start_trim = clip_selection.kwargs["clips_range"][0]
            end_trim = clip_selection.kwargs["clips_range"][1]

            # Calculate all the seconds for the new clips to start
            movie_i_df["seconds"] = [
                list(
                    range(
                        start_trim,
                        int(math.floor(end_trim / clip_length) * clip_length),
                        clip_length,
                    )
                )
            ]
        else:
            movie_i_df["seconds"] = [[0]]

        # Reshape the dataframe with the seconds for the new clips to start on the rows
        potential_start_df = t_utils.expand_list(movie_i_df, "seconds", "upl_seconds")

        # Specify the length of the clips
        potential_start_df["clip_length"] = clip_length

        if not clip_numbers == potential_start_df.shape[0]:
            logging.info(
                f"There was an issue estimating the starting seconds for the {clip_numbers} clips"
            )

        # Get project-specific server info
        server = self.project.server

        # Specify the temp folder to host the clips
        temp_clip_folder = movie_i + "_zooniverseclips"
        if server == "SNIC":
            snic_path = "/mimer/NOBACKUP/groups/snic2021-6-9/"
            clips_folder = Path(snic_path, "tmp_dir", temp_clip_folder)
        else:
            clips_folder = temp_clip_folder

        # Set the filename of the clips
        potential_start_df["clip_filename"] = (
            movie_i
            + "_clip_"
            + potential_start_df["upl_seconds"].astype(str)
            + "_"
            + str(clip_length)
            + ".mp4"
        )

        # Set the path of the clips
        potential_start_df["clip_path"] = potential_start_df["clip_filename"].apply(
            lambda x: Path(clips_folder, x), 1
        )

        # Create the folder to store the videos if not exist
        if os.path.exists(clips_folder):
            shutil.rmtree(clips_folder)
        Path(clips_folder).mkdir(parents=True, exist_ok=True)
        # Recursively add permissions to folders created
        [os.chmod(root, 0o777) for root, dirs, files in os.walk(clips_folder)]

        logging.info("Extracting clips")

        # Read each movie and extract the clips
        for index, row in potential_start_df.iterrows():
            # Extract the videos and store them in the folder
            t_utils.extract_clips(
                movie_path,
                clip_length,
                row["upl_seconds"],
                row["clip_path"],
                modification_details,
                gpu_available,
            )

        # Add information on the modification of the clips
        potential_start_df["clip_modification_details"] = str(modification_details)
        return potential_start_df

    def set_zoo_metadata(
        self,
        df: pd.DataFrame,
    ):
        """
        It takes a dataframe with clips, and adds metadata about the site and project to it

        :param df: the dataframe with the clips to upload
        :return: upload_to_zoo, sitename, created_on
        """

        # Create connection to db
        conn = db_utils.create_connection(self.db_info["db_path"])

        # Query info about the movie of interest
        sitesdf = pd.read_sql_query("SELECT * FROM sites", conn)

        # Rename the id column to match movies df
        sitesdf = sitesdf.rename(
            columns={
                "id": "site_id",
            }
        )

        # Combine site info to the df
        if "site_id" in df.columns:
            upload_to_zoo = df.merge(sitesdf, on="site_id")
            sitename = upload_to_zoo["siteName"].unique()[0]
        else:
            raise ValueError(
                "Sites table empty. Perhaps try to rebuild the initial db."
            )

        # Rename columns to match schema names
        # (fields that begin with “#” or “//” will never be shown to volunteers)
        # (fields that begin with "!" will only be available for volunteers on the Talk section, after classification)
        upload_to_zoo = upload_to_zoo.rename(
            columns={
                "id": "movie_id",
                "created_on": "#created_on",
                "clip_length": "#clip_length",
                "filename": "#VideoFilename",
                "clip_modification_details": "#clip_modification_details",
                "siteName": "#siteName",
            }
        )

        # Convert datetime to string to avoid JSON seriazible issues
        upload_to_zoo["#created_on"] = upload_to_zoo["#created_on"].astype(str)
        created_on = upload_to_zoo["#created_on"].unique()[0]

        # Select only relevant columns
        upload_to_zoo = upload_to_zoo[
            [
                "movie_id",
                "clip_path",
                "upl_seconds",
                "#clip_length",
                "#created_on",
                "#VideoFilename",
                "#siteName",
                "#clip_modification_details",
            ]
        ]

        # Add information about the type of subject
        upload_to_zoo["Subject_type"] = "clip"

        # Add spyfish-specific info
        if self.project.Project_name == "Spyfish_Aotearoa":
            # Read sites csv as pd
            sitesdf = pd.read_csv(self.db_info["local_sites_csv"])

            # Read movies csv as pd
            moviesdf = pd.read_csv(self.db_info["local_movies_csv"])

            # Rename columns to match schema names
            sitesdf = sitesdf.rename(
                columns={
                    "siteName": "SiteID",
                }
            )

            # Include movie info to the sites df
            sitesdf = sitesdf.merge(moviesdf, on="SiteID")

            # Rename columns to match schema names
            sitesdf = sitesdf.rename(
                columns={
                    "LinkToMarineReserve": "!LinkToMarineReserve",
                    "SiteID": "#SiteID",
                }
            )

            # Select only relevant columns
            sitesdf = sitesdf[["!LinkToMarineReserve", "#SiteID", "ProtectionStatus"]]

            # Include site info to the df
            upload_to_zoo = upload_to_zoo.merge(
                sitesdf, left_on="#siteName", right_on="#SiteID"
            )

        if self.project.Project_name == "Koster_Seafloor_Obs":
            # Read sites csv as pd
            sitesdf = pd.read_csv(self.db_info["local_sites_csv"])

            # Rename columns to match schema names
            sitesdf = sitesdf.rename(
                columns={
                    "decimalLatitude": "#decimalLatitude",
                    "decimalLongitude": "#decimalLongitude",
                    "geodeticDatum": "#geodeticDatum",
                    "countryCode": "#countryCode",
                }
            )

            # Select only relevant columns
            sitesdf = sitesdf[
                [
                    "siteName",
                    "#decimalLatitude",
                    "#decimalLongitude",
                    "#geodeticDatum",
                    "#countryCode",
                ]
            ]

            # Include site info to the df
            upload_to_zoo = upload_to_zoo.merge(
                sitesdf, left_on="#siteName", right_on="siteName"
            )

        # Prevent NANs on any column
        if upload_to_zoo.isnull().values.any():
            logging.info(
                f"The following columns have NAN values {upload_to_zoo.columns[upload_to_zoo.isna().any()].tolist()}"
            )

        logging.info(
            f"The metadata for the {upload_to_zoo.shape[0]} subjects is ready."
        )

        return upload_to_zoo, sitename, created_on

    def upload_clips_to_zooniverse(
        self,
        upload_to_zoo: pd.DataFrame,
        sitename: str,
        created_on: str,
    ):
        """
        It takes a dataframe of clips and metadata, creates a new subject set, and uploads the clips to
        Zooniverse

        :param upload_to_zoo: the dataframe of clips to upload
        :param sitename: the name of the site you're uploading clips from
        :param created_on: the date the clips were created
        :param project: the project ID of the project you want to upload to
        """

        # Estimate the number of clips
        n_clips = upload_to_zoo.shape[0]

        # Create a new subject set to host the clips
        subject_set = SubjectSet()
        subject_set_name = (
            "clips_" + sitename + "_" + str(int(n_clips)) + "_" + created_on
        )
        subject_set.links.project = self.project.Zooniverse_number
        subject_set.display_name = subject_set_name

        subject_set.save()

        logging.info(f"{subject_set_name} subject set created")

        # Save the df as the subject metadata
        subject_metadata = upload_to_zoo.set_index("clip_path").to_dict("index")

        # Upload the clips to Zooniverse (with metadata)
        new_subjects = []

        logging.info("Uploading subjects to Zooniverse")
        for modif_clip_path, metadata in tqdm(
            subject_metadata.items(), total=len(subject_metadata)
        ):
            # Create a subject
            subject = Subject()

            # Add project info
            subject.links.project = self.project.Zooniverse_number

            # Add location of clip
            subject.add_location(modif_clip_path)

            # Add metadata
            subject.metadata.update(metadata)

            # Save subject info
            subject.save()
            new_subjects.append(subject)

        # Upload all subjects
        subject_set.add(new_subjects)

        logging.info("Subjects uploaded to Zooniverse")

    def check_movie_uploaded(self, movie_i: str):
        """
        This function takes in a movie name and a dictionary containing the path to the database and returns
        a boolean value indicating whether the movie has already been uploaded to Zooniverse

        :param movie_i: the name of the movie you want to check
        :type movie_i: str
        :param db_info_dict: a dictionary containing the path to the database and the path to the folder containing the videos
        :type db_info_dict: dict
        """

        # Create connection to db
        conn = db_utils.create_connection(self.db_info["db_path"])

        # Query info about the clip subjects uploaded to Zooniverse
        subjects_df = pd.read_sql_query(
            "SELECT id, subject_type, filename, clip_start_time,"
            "clip_end_time, movie_id FROM subjects WHERE subject_type='clip'",
            conn,
        )

        # Save the video filenames of the clips uploaded to Zooniverse
        videos_uploaded = subjects_df.filename.dropna().unique()

        # Check if selected movie has already been uploaded
        already_uploaded = any(mv in movie_i for mv in videos_uploaded)

        if already_uploaded:
            clips_uploaded = subjects_df[subjects_df["filename"].str.contains(movie_i)]
            logging.info(f"{movie_i} has clips already uploaded.")
            logging.info(clips_uploaded.head())
        else:
            logging.info(f"{movie_i} has not been uploaded to Zooniverse yet")

    def generate_zu_clips(
        self,
        movie_name,
        movie_path,
        use_gpu: bool = False,
        pool_size: int = 4,
        is_example: bool = False,
    ):
        """
        > This function takes a movie name and path, and returns a list of clips from that movie

        :param movie_name: The name of the movie you want to extract clips from
        :param movie_path: The path to the movie you want to extract clips from
        :param use_gpu: If you have a GPU, set this to True, defaults to False
        :type use_gpu: bool (optional)
        :param pool_size: number of threads to use for clip extraction, defaults to 4
        :type pool_size: int (optional)
        :param is_example: If True, the clips will be selected randomly. If False, the clips will be
               selected based on the number of clips and the length of each clip, defaults to False
        :type is_example: bool (optional)
        """
        # t3_utils.create_clips

        if is_example:
            clip_selection = self.select_random_clips(movie_i=movie_name)
        else:
            clip_selection = self.select_clip_n_len(movie_i=movie_name)

        clip_modification = t_utils.clip_modification_widget()

        button = widgets.Button(
            description="Click to extract clips.",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
        )

        def on_button_clicked(b):
            self.generated_clips = self.create_clips(
                self.server_movies_csv,
                movie_name,
                movie_path,
                clip_selection,
                {},
                use_gpu,
                pool_size,
            )
            mod_clips = self.create_modified_clips(
                self.generated_clips.clip_path,
                movie_name,
                clip_modification.checks,
                use_gpu,
                pool_size,
            )
            # Temporary workaround to get both clip paths
            self.generated_clips["modif_clip_path"] = mod_clips

        button.on_click(on_button_clicked)
        display(clip_modification)
        display(button)

    def create_modified_clips(
        self,
        clips_list: list,
        movie_i: str,
        modification_details: dict,
        gpu_available: bool,
        pool_size: int = 4,
    ):
        """
        This function takes a list of clips, a movie name, a dictionary of modifications, a project, and a
        GPU availability flag, and returns a list of modified clips

        :param clips_list: a list of the paths to the clips you want to modify
        :param movie_i: the path to the movie you want to extract clips from
        :param modification_details: a dictionary with the modifications to be applied to the clips. The keys are the names of the modifications and the values are the parameters of the modifications
        :param project: the project object
        :param gpu_available: True if you have a GPU available, False if you don't
        :param pool_size: the number of parallel processes to run, defaults to 4 (optional)
        :return: The modified clips
        """

        # Get project-specific server info
        server = self.project.server

        # Specify the folder to host the modified clips

        mod_clip_folder = "modified_" + movie_i + "_clips"

        if server == "SNIC":
            snic_path = "/mimer/NOBACKUP/groups/snic2021-6-9/"
            mod_clips_folder = Path(snic_path, "tmp_dir", mod_clip_folder)
        else:
            mod_clips_folder = mod_clip_folder

        # Remove existing modified clips
        if os.path.exists(mod_clips_folder):
            shutil.rmtree(mod_clips_folder)

        if len(modification_details.values()) > 0:
            # Create the folder to store the videos if not exist
            if not os.path.exists(mod_clips_folder):
                Path(mod_clips_folder).mkdir(parents=True, exist_ok=True)
                # Recursively add permissions to folders created
                [
                    os.chmod(root, 0o777)
                    for root, dirs, files in os.walk(mod_clips_folder)
                ]

            # Specify the number of parallel items
            pool = multiprocessing.Pool(pool_size)

            # Create empty list to keep track of new clips
            modified_clips = []
            results = []
            # Create the information for each clip and extract it (printing a progress bar)
            for clip_i in clips_list:
                # Create the filename and path of the modified clip
                output_clip_name = "modified_" + os.path.basename(clip_i)
                output_clip_path = Path(mod_clips_folder, output_clip_name)

                # Add the path of the clip to the list
                modified_clips = modified_clips + [output_clip_path]

                # Modify the clips and store them in the folder
                results.append(
                    pool.apply_async(
                        t_utils.modify_clips,
                        (
                            clip_i,
                            modification_details,
                            output_clip_path,
                            gpu_available,
                        ),
                    )
                )

            pool.close()
            pool.join()
            return modified_clips
        else:
            logging.info("No modification selected")

    def check_movies_uploaded(self, movie_name: str):
        """
        This function checks if a movie has been uploaded to Zooniverse

        :param movie_name: The name of the movie you want to check if it's uploaded
        :type movie_name: str
        """
        self.check_movie_uploaded(self, movie_i=movie_name)

    def choose_species(self):
        """
        This function generates a widget to select the species of interest
        :param db_info_dict: a dictionary containing the path to the database
        :type db_info_dict: dict
        """
        # Create connection to db
        conn = db_utils.create_connection(self.db_info["db_path"])

        # Get a list of the species available
        species_list = pd.read_sql_query("SELECT label from species", conn)[
            "label"
        ].tolist()

        # Roadblock to check if species list is empty
        if len(species_list) == 0:
            raise ValueError(
                "Your database contains no species, please add at least one species before continuing."
            )

        # Generate the widget
        w = widgets.SelectMultiple(
            options=species_list,
            value=[species_list[0]],
            description="Species",
            disabled=False,
        )

        display(w)
        return w

    # Function to match species selected to species id
    def get_species_ids(self, species_list: list):
        """
        # Get ids of species of interest
        """
        db_path = self.project.db_path
        conn = db_utils.create_connection(db_path)
        if len(species_list) == 1:
            species_ids = pd.read_sql_query(
                f'SELECT id FROM species WHERE label=="{species_list[0]}"', conn
            )["id"].tolist()
        else:
            species_ids = pd.read_sql_query(
                f"SELECT id FROM species WHERE label IN {tuple(species_list)}", conn
            )["id"].tolist()
        return species_ids

    def get_species_frames(
        self,
        agg_clips_df: pd.DataFrame,
        species_ids: list,
        n_frames_subject: int,
    ):
        """
        # Function to identify up to n number of frames per classified clip
        # that contains species of interest after the first time seen

        # Find classified clips that contain the species of interest
        """

        # Retrieve list of subjects
        subjects_df = pd.read_sql_query(
            "SELECT id, clip_start_time, movie_id FROM subjects WHERE subject_type='clip'",
            self.db_connection,
        )

        agg_clips_df["subject_ids"] = pd.to_numeric(
            agg_clips_df["subject_ids"], errors="coerce"
        ).astype("Int64")
        subjects_df["id"] = pd.to_numeric(subjects_df["id"], errors="coerce").astype(
            "Int64"
        )

        # Combine the aggregated clips and subjects dataframes
        frames_df = pd.merge(
            agg_clips_df, subjects_df, how="left", left_on="subject_ids", right_on="id"
        ).drop(columns=["id"])

        # Identify the second of the original movie when the species first appears
        frames_df["first_seen_movie"] = (
            frames_df["clip_start_time"] + frames_df["first_seen"]
        )

        server = self.project.server

        if server in ["SNIC", "TEMPLATE"]:
            movies_df = movie_utils.retrieve_movie_info_from_server(
                self.project, self.server_dict
            )

            # Include movies' filepath and fps to the df
            frames_df = frames_df.merge(
                movies_df, left_on="movie_id", right_on="movie_id"
            )
            frames_df["fpath"] = frames_df["spath"]

        if len(frames_df[~frames_df.exists]) > 0:
            logging.error(
                f"There are {len(frames_df) - frames_df.exists.sum()} out of {len(frames_df)} frames with a missing movie"
            )

        # Select only frames from movies that can be found
        frames_df = frames_df[frames_df.exists]
        if len(frames_df) == 0:
            logging.error(
                "There are no frames for this species that meet your aggregation criteria."
                "Please adjust your aggregation criteria / species choice and try again."
            )

        ##### Add species_id info ####
        # Retrieve species info
        species_df = pd.read_sql_query(
            "SELECT id, label, scientificName FROM species",
            self.db_connection,
        )

        # Retrieve species info
        species_df = species_df.rename(columns={"id": "species_id"})

        # Match format of species name to Zooniverse labels
        species_df["label"] = species_df["label"].apply(t_utils.clean_label)

        # Combine the aggregated clips and subjects dataframes
        frames_df = pd.merge(frames_df, species_df, how="left", on="label")

        # Identify the ordinal number of the frames expected to be extracted
        if len(frames_df) == 0:
            raise ValueError("No frames. Workflow stopped.")

        frames_df["frame_number"] = frames_df[["first_seen_movie", "fps"]].apply(
            lambda x: [
                int((x["first_seen_movie"] + j) * x["fps"])
                for j in range(n_frames_subject)
            ],
            1,
        )

        # Reshape df to have each frame as rows
        lst_col = "frame_number"

        frames_df = pd.DataFrame(
            {
                col: np.repeat(frames_df[col].values, frames_df[lst_col].str.len())
                for col in frames_df.columns.difference([lst_col])
            }
        ).assign(**{lst_col: np.concatenate(frames_df[lst_col].values)})[
            frames_df.columns.tolist()
        ]

        # Drop unnecessary columns
        frames_df.drop(["subject_ids"], inplace=True, axis=1)

        return frames_df

    # Function to gather information of frames already uploaded
    def check_frames_uploaded(
        self,
        frames_df: pd.DataFrame,
        species_ids: list,
    ):
        if self.project.server == "SNIC":
            # Get info of frames of the species of interest already uploaded
            if len(species_ids) <= 1:
                uploaded_frames_df = pd.read_sql_query(
                    f"SELECT movie_id, frame_number, \
                frame_exp_sp_id FROM subjects WHERE frame_exp_sp_id=='{species_ids[0]}' AND subject_type='frame'",
                    self.db_connection,
                )

            else:
                uploaded_frames_df = pd.read_sql_query(
                    f"SELECT movie_id, frame_number, frame_exp_sp_id FROM subjects WHERE frame_exp_sp_id IN \
                {tuple(species_ids)} AND subject_type='frame'",
                    self.db_connection,
                )

            # Filter out frames that have already been uploaded
            if (
                len(uploaded_frames_df) > 0
                and not uploaded_frames_df["frame_number"].isnull().any()
            ):
                logging.info(
                    "There are some frames already uploaded in Zooniverse for the species selected. \
                    Checking if those are the frames you are trying to upload"
                )
                # Ensure that frame_number is an integer
                uploaded_frames_df["frame_number"] = uploaded_frames_df[
                    "frame_number"
                ].astype(int)
                frames_df["frame_number"] = frames_df["frame_number"].astype(int)
                merge_df = (
                    pd.merge(
                        frames_df,
                        uploaded_frames_df,
                        left_on=["movie_id", "frame_number"],
                        right_on=["movie_id", "frame_number"],
                        how="left",
                        indicator=True,
                    )["_merge"]
                    == "both"
                )

                # Exclude frames that have already been uploaded
                # trunk-ignore(flake8/E712)
                frames_df = frames_df[merge_df == False]
                if len(frames_df) == 0:
                    logging.error(
                        "All of the frames you have selected are already uploaded."
                    )
                else:
                    logging.info(
                        "There are",
                        len(frames_df),
                        "frames with the species of interest not uploaded to Zooniverse yet.",
                    )

            else:
                logging.info(
                    "There are no frames uploaded in Zooniverse for the species selected."
                )

        return frames_df

    # Function to extract selected frames from videos
    def extract_frames(
        self,
        df: pd.DataFrame,
        frames_folder: str,
    ):
        """
        Extract frames and save them in chosen folder.
        """

        # Set the filename of the frames
        df["frame_path"] = (
            frames_folder
            + df["filename"].astype(str)
            + "_frame_"
            + df["frame_number"].astype(str)
            + "_"
            + df["label"].astype(str)
            + ".jpg"
        )

        # Create the folder to store the frames if not exist
        if not os.path.exists(frames_folder):
            Path(frames_folder).mkdir(parents=True, exist_ok=True)
            # Recursively add permissions to folders created
            [os.chmod(root, 0o777) for root, dirs, files in os.walk(frames_folder)]

        for movie in df["fpath"].unique():
            url = movie_utils.get_movie_path(
                project=self.project, db_info_dict=self.server_info, f_path=movie
            )

            if url is None:
                logging.error(f"Movie {movie} couldn't be found in the server.")
            else:
                # Select the frames to download from the movie
                key_movie_df = df[df["fpath"] == movie].reset_index()

                # Read the movie on cv2 and prepare to extract frames
                t_utils.write_movie_frames(key_movie_df, url)

            logging.info("Frames extracted successfully")

        return df

    def process_clips(self, df: pd.DataFrame):
        """
        This function takes a dataframe of classifications and returns a dataframe of annotations

        :param df: the dataframe of classifications
        :type df: pd.DataFrame
        :return: A dataframe with the classification_id, label, how_many, first_seen, https_location,
                subject_type, and subject_ids.
        """

        # Create an empty list
        rows_list = []

        # Loop through each classification submitted by the users
        for index, row in df.iterrows():
            # Load annotations as json format
            annotations = json.loads(row["annotations"])

            # Select the information from the species identification task
            if self.project.Zooniverse_number == 9747:
                rows_list = process_clips_koster(
                    annotations, row["classification_id"], rows_list
                )

            # Check if the Zooniverse project is the Spyfish
            if self.project.Project_name == "Spyfish_Aotearoa":
                rows_list = process_clips_spyfish(
                    annotations, row["classification_id"], rows_list
                )

            # Process clips as the default method
            else:
                rows_list = zu_utils.process_clips_template(
                    annotations, row["classification_id"], rows_list
                )

        # Create a data frame with annotations as rows
        annot_df = pd.DataFrame(
            rows_list, columns=["classification_id", "label", "first_seen", "how_many"]
        )

        # Specify the type of columns of the df
        annot_df["how_many"] = pd.to_numeric(annot_df["how_many"])
        annot_df["first_seen"] = pd.to_numeric(annot_df["first_seen"])

        # Add subject id to each annotation
        annot_df = pd.merge(
            annot_df,
            df.drop(columns=["annotations"]),
            how="left",
            on="classification_id",
        )

        # Select only relevant columns
        annot_df = annot_df[
            [
                "classification_id",
                "label",
                "how_many",
                "first_seen",
                "https_location",
                "subject_type",
                "subject_ids",
                "workflow_id",
                "workflow_name",
                "workflow_version",
            ]
        ]

        return pd.DataFrame(annot_df)

    def aggregate_classifications(
        self, df: pd.DataFrame, subj_type: str, agg_params: list
    ):
        """
        We take the raw classifications and process them to get the aggregated labels

        :param df: the raw classifications dataframe
        :param subj_type: the type of subject, either "frame" or "clip"
        :param agg_params: list of parameters for the aggregation
        :return: the aggregated classifications and the raw classifications.
        """

        logging.info("Aggregating the classifications")

        # We take the raw classifications and process them to get the aggregated labels.
        if subj_type == "frame":
            # Get the aggregation parameters
            if not isinstance(agg_params, list):
                agg_users, min_users, agg_obj, agg_iou, agg_iua = [
                    i.value for i in agg_params
                ]
            else:
                agg_users, min_users, agg_obj, agg_iou, agg_iua = agg_params

            # Report selected parameters
            logging.info(
                f"Aggregation parameters are: Agg. threshold "
                f"{agg_users} "
                f"Min. users "
                f"{min_users} "
                f"Obj threshold "
                f"{agg_obj} "
                f"IOU "
                f"{agg_iou} "
                f"Int. agg. "
                f"{agg_iua} "
            )
            # Process the raw classifications
            raw_class_df = self.process_frames(df, self.project.Project_name)

            # Aggregate frames based on their labels
            agg_labels_df = t_utils.aggregate_labels(raw_class_df, agg_users, min_users)

            # Get rid of the "empty" labels if other species are among the volunteer consensus
            agg_labels_df = agg_labels_df[
                ~(
                    (agg_labels_df["class_n_agg"] > 1)
                    & (agg_labels_df["label"] == "empty")
                )
            ]

            # Select frames aggregated only as empty
            agg_labels_df_empty = agg_labels_df[agg_labels_df["label"] == "empty"]
            agg_labels_df_empty = agg_labels_df_empty.rename(
                columns={"frame_number": "start_frame"}
            )
            agg_labels_df_empty = agg_labels_df_empty[
                ["label", "subject_ids", "x", "y", "w", "h"]
            ]

            # Temporary exclude frames aggregated as empty
            agg_labels_df = agg_labels_df[agg_labels_df["label"] != "empty"]

            # Map the position of the annotation parameters
            col_list = list(agg_labels_df.columns)
            x_pos, y_pos, w_pos, h_pos, user_pos, subject_id_pos = (
                col_list.index("x"),
                col_list.index("y"),
                col_list.index("w"),
                col_list.index("h"),
                col_list.index("user_name"),
                col_list.index("subject_ids"),
            )

            # Get prepared annotations
            new_rows = []

            if agg_labels_df["frame_number"].isnull().all():
                group_cols = ["subject_ids", "label"]
            else:
                group_cols = ["subject_ids", "label", "frame_number"]

            for name, group in agg_labels_df.groupby(group_cols):
                if "frame_number" in group_cols:
                    subj_id, label, start_frame = name
                    total_users = agg_labels_df[
                        (agg_labels_df.subject_ids == subj_id)
                        & (agg_labels_df.label == label)
                        & (agg_labels_df.frame_number == start_frame)
                    ]["user_name"].nunique()
                else:
                    subj_id, label = name
                    start_frame = np.nan
                    total_users = agg_labels_df[
                        (agg_labels_df.subject_ids == subj_id)
                        & (agg_labels_df.label == label)
                    ]["user_name"].nunique()

                # Filter bboxes using IOU metric (essentially a consensus metric)
                # Keep only bboxes where mean overlap exceeds this threshold
                indices, new_group = filter_bboxes(
                    total_users=total_users,
                    users=[i[user_pos] for i in group.values],
                    bboxes=[
                        np.array([i[x_pos], i[y_pos], i[w_pos], i[h_pos]])
                        for i in group.values
                    ],
                    obj=agg_obj,
                    eps=agg_iou,
                    iua=agg_iua,
                )

                subject_ids = [i[subject_id_pos] for i in group.values[indices]]

                for ix, box in zip(subject_ids, new_group):
                    new_rows.append(
                        (
                            label,
                            start_frame,
                            ix,
                        )
                        + tuple(box)
                    )

            agg_class_df = pd.DataFrame(
                new_rows,
                columns=[
                    "label",
                    "start_frame",
                    "subject_ids",
                    "x",
                    "y",
                    "w",
                    "h",
                ],
            )

            agg_class_df["subject_type"] = "frame"
            agg_class_df["label"] = agg_class_df["label"].apply(
                lambda x: x.split("(")[0].strip()
            )

            # Add the frames aggregated as "empty"
            agg_class_df = pd.concat([agg_class_df, agg_labels_df_empty])

            # Select the aggregated labels
            agg_class_df = agg_class_df[
                ["subject_ids", "label", "x", "y", "w", "h"]
            ].drop_duplicates()

            # Add the http info
            agg_class_df = pd.merge(
                agg_class_df,
                raw_class_df[
                    [
                        "subject_ids",
                        "https_location",
                        "subject_type",
                        "workflow_id",
                        "workflow_name",
                        "workflow_version",
                    ]
                ].drop_duplicates(),
                how="left",
                on="subject_ids",
            )

        else:
            # Get the aggregation parameters
            if not isinstance(agg_params, list):
                agg_users, min_users = [i.value for i in agg_params]
            else:
                agg_users, min_users = agg_params

            # Process the raw classifications
            raw_class_df = self.process_clips(df)

            # aggregate clips based on their labels
            agg_class_df = t_utils.aggregate_labels(raw_class_df, agg_users, min_users)

            # Extract the median of the second where the animal/object is and number of animals
            agg_class_df = agg_class_df.groupby(
                [
                    "subject_ids",
                    "https_location",
                    "subject_type",
                    "label",
                    "workflow_id",
                    "workflow_name",
                    "workflow_version",
                ],
                as_index=False,
            )
            agg_class_df = pd.DataFrame(
                agg_class_df[["how_many", "first_seen"]].median().round(0)
            )

        # Add username info to raw class
        raw_class_df = pd.merge(
            raw_class_df,
            df[["classification_id", "user_name"]],
            how="left",
            on="classification_id",
        )

        logging.info(
            f"{agg_class_df.shape[0]}"
            "classifications aggregated out of"
            f"{df.subject_ids.nunique()}"
            "unique subjects available"
        )

        return agg_class_df, raw_class_df

    # Function to the provide drop-down options to select the frames to be uploaded
    def get_frames(
        self,
        species_names: list,
        n_frames_subject=3,
        subsample_up_to=100,
    ):
        # Roadblock to check if species list is empty
        if len(species_names) == 0:
            raise ValueError(
                "No species were selected. Please select at least one species before continuing."
            )

        # Transform species names to species ids
        species_ids = self.get_species_ids(species_names)

        conn = db_utils.create_connection(self.project.db_path)

        if self.project.movie_folder is None:
            # Extract frames of interest from a folder with frames
            if self.project.server == "SNIC":
                # Specify volume allocated by SNIC
                snic_path = "/mimer/NOBACKUP/groups/snic2021-6-9"
                df = FileChooser(str(Path(snic_path, "tmp_dir")))
            else:
                df = FileChooser(".")
            df.title = "<b>Select frame folder location</b>"

            # Callback function
            def build_df(chooser):
                frame_files = os.listdir(chooser.selected)
                frame_paths = [os.path.join(chooser.selected, i) for i in frame_files]
                chooser.df = pd.DataFrame(frame_paths, columns=["frame_path"])

                if isinstance(species_ids, list):
                    chooser.df["species_id"] = str(species_ids)
                else:
                    chooser.df["species_id"] = species_ids

            # Register callback function
            df.register_callback(build_df)
            display(df)

        else:
            ## Choose the Zooniverse workflow/s with classified clips to extract the frames from ####
            # Select the Zooniverse workflow/s of interest
            workflows_out = t_utils.WidgetMaker(self.zoo_info["workflows"])
            display(workflows_out)

            # Select the agreement threshold to aggregrate the responses
            agg_params = t_utils.choose_agg_parameters("clip")

            # Select the temp location to store frames before uploading them to Zooniverse
            if self.project.server == "SNIC":
                # Specify volume allocated by SNIC
                snic_path = "/mimer/NOBACKUP/groups/snic2021-6-9"
                df = FileChooser(str(Path(snic_path, "tmp_dir")))
            else:
                df = FileChooser(".")
            df.title = "<b>Choose location to store frames</b>"

            # Callback function
            def extract_files(chooser):
                # Get the aggregated classifications based on the specified agreement threshold
                clips_df = self.get_classifications(
                    workflows_out.checks,
                    self.zoo_info["workflows"],
                    "clip",
                    self.zoo_info["classifications"],
                    self.project.db_path,
                )

                agg_clips_df, raw_clips_df = self.aggregate_classifications(
                    clips_df, "clip", agg_params=agg_params
                )

                # Match format of species name to Zooniverse labels
                species_names_zoo = [
                    t_utils.clean_label(species_name) for species_name in species_names
                ]

                # Select only aggregated classifications of species of interest:
                sp_agg_clips_df = agg_clips_df[
                    agg_clips_df["label"].isin(species_names_zoo)
                ]

                # Subsample up to desired sample
                if sp_agg_clips_df.shape[0] >= subsample_up_to:
                    logging.info("Subsampling up to " + str(subsample_up_to))
                    sp_agg_clips_df = sp_agg_clips_df.sample(subsample_up_to)

                # Populate the db with the aggregated classifications
                zu_utils.populate_agg_annotations(sp_agg_clips_df, "clip", self.project)

                # Get df of frames to be extracted
                frame_df = self.get_species_frames(
                    sp_agg_clips_df,
                    species_ids,
                    conn,
                    n_frames_subject,
                )

                # Check the frames haven't been uploaded to Zooniverse
                frame_df = self.check_frames_uploaded(frame_df, species_ids, conn)

                # Extract the frames from the videos and store them in the temp location
                if self.project.server == "SNIC":
                    folder_name = chooser.selected
                    frames_folder = Path(
                        folder_name, "_".join(species_names_zoo) + "_frames/"
                    )
                else:
                    frames_folder = "_".join(species_names_zoo) + "_frames/"
                chooser.df = self.extract_frames(frame_df, frames_folder)

            # Register callback function
            df.register_callback(extract_files)
            display(df)

        return df

    def upload_zu_subjects(self, upload_data: pd.DataFrame, subject_type: str):
        """
        This function uploads clips or frames to Zooniverse, depending on the subject_type argument

        :param upload_data: a pandas dataframe with the following columns:
        :type upload_data: pd.DataFrame
        :param subject_type: str = "clip" or "frame"
        :type subject_type: str
        """
        if subject_type == "clip":
            upload_df, sitename, created_on = self.set_zoo_metadata(upload_data)
            self.upload_clips_to_zooniverse(upload_df, sitename, created_on)
            # Clean up subjects after upload
            t_utils.remove_temp_clips(upload_df)
        elif subject_type == "frame":
            species_list = []
            upload_df = self.set_zoo_metadata(upload_data, species_list)
            self.upload_frames_to_zooniverse(upload_df, species_list)

    # Function to set the metadata of the frames to be uploaded to Zooniverse
    def set_zoo_metadata(self, df, species_list: list):
        project_name = self.project.Project_name

        if not isinstance(df, pd.DataFrame):
            df = df.df

        if (
            "modif_frame_path" in df.columns
            and "no_modification" not in df["modif_frame_path"].values
        ):
            df["frame_path"] = df["modif_frame_path"]

        # Set project-specific metadata
        if self.project.Zooniverse_number == 9747 or 9754:
            conn = db_utils.create_connection(self.project.db_path)
            sites_df = pd.read_sql_query("SELECT id, siteName FROM sites", conn)
            df = df.merge(sites_df, left_on="site_id", right_on="id")
            upload_to_zoo = df[
                [
                    "frame_path",
                    "frame_number",
                    "species_id",
                    "movie_id",
                    "created_on",
                    "siteName",
                ]
            ]

        elif project_name == "SGU":
            upload_to_zoo = df[["frame_path", "species_id", "filename"]]

        elif project_name == "Spyfish_Aotearoa":
            upload_to_zoo = spyfish_subject_metadata(df, self.db_info)
        else:
            logging.error("This project is not a supported Zooniverse project.")

        # Add information about the type of subject
        upload_to_zoo = upload_to_zoo.copy()
        upload_to_zoo.loc[:, "subject_type"] = "frame"
        upload_to_zoo = upload_to_zoo.rename(columns={"species_id": "frame_exp_sp_id"})

        # Check there are no empty values (prevent issues uploading subjects)
        if upload_to_zoo.isnull().values.any():
            logging.error(
                "There are some values missing from the data you are trying to upload."
            )

        return upload_to_zoo

    # Function to upload frames to Zooniverse
    def upload_frames_to_zooniverse(
        self,
        upload_to_zoo: dict,
        species_list: list,
    ):
        # Retireve zooniverse project name and number
        project_name = self.project.Project_name
        project_number = self.project.Zooniverse_number

        # Estimate the number of frames
        n_frames = upload_to_zoo.shape[0]

        if project_name == "Koster_Seafloor_Obs":
            created_on = upload_to_zoo["created_on"].unique()[0]
            sitename = upload_to_zoo["siteName"].unique()[0]

            # Name the subject set
            subject_set_name = (
                "frames_"
                + str(int(n_frames))
                + "_"
                + "_".join(species_list)
                + "_"
                + sitename
                + "_"
                + created_on
            )

        elif project_name == "SGU":
            surveys_df = pd.read_csv(self.db_info["local_surveys_csv"])
            created_on = surveys_df["SurveyDate"].unique()[0]
            folder_name = os.path.split(
                os.path.dirname(upload_to_zoo["frame_path"].iloc[0])
            )[1]
            sitename = folder_name

            # Name the subject set
            subject_set_name = (
                "frames_"
                + str(int(n_frames))
                + "_"
                + "_".join(species_list)
                + "_"
                + sitename
                + "_"
                + created_on
            )

        else:
            # Name the subject for frames from multiple sites/movies
            subject_set_name = (
                "frames_"
                + str(int(n_frames))
                + "_"
                + "_".join(species_list)
                + datetime.date.today().strftime("_%d_%m_%Y")
            )

        # Create a new subject set to host the frames
        subject_set = SubjectSet()
        subject_set.links.project = project_number
        subject_set.display_name = subject_set_name
        subject_set.save()

        logging.info(subject_set_name, "subject set created")

        # Save the df as the subject metadata
        subject_metadata = upload_to_zoo.set_index("frame_path").to_dict("index")

        # Upload the clips to Zooniverse (with metadata)
        new_subjects = []

        logging.info("Uploading subjects to Zooniverse...")
        for frame_path, metadata in tqdm(
            subject_metadata.items(), total=len(subject_metadata)
        ):
            subject = Subject()

            subject.links.project = project_number
            subject.add_location(frame_path)

            logging.info(frame_path)
            subject.metadata.update(metadata)

            logging.info(metadata)
            subject.save()
            logging.info("Subject saved")
            new_subjects.append(subject)

        # Upload videos
        subject_set.add(new_subjects)
        logging.info("Subjects uploaded to Zooniverse")

    def generate_zu_frames(self):
        """
        This function takes a dataframe of frames to upload, a species of interest, a project, and a
        dictionary of modifications to make to the frames, and returns a dataframe of modified frames.
        """

        frame_modification = t_utils.clip_modification_widget()

        button = widgets.Button(
            description="Click to modify frames",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
        )

        def on_button_clicked(b):
            self.generated_frames = self.modify_frames(
                frames_to_upload_df=self.frames_to_upload_df.df.reset_index(drop=True),
                species_i=self.species_of_interest,
                modification_details=frame_modification.checks,
            )

        button.on_click(on_button_clicked)
        display(frame_modification)
        display(button)

    def generate_custom_frames(
        self,
        input_path: str,
        output_path: str,
        num_frames: int = None,
        frames_skip: int = None,
    ):
        """
        This function generates custom frames from input movie files and saves them in an output directory.

        :param input_path: The directory path where the input movie files are located
        :type input_path: str
        :param output_path: The directory where the extracted frames will be saved
        :type output_path: str
        :param num_frames: The number of frames to extract from each video file. If not specified, all
        frames will be extracted
        :type num_frames: int
        :param frames_skip: The `frames_skip` parameter is an optional integer that specifies the number of
        frames to skip between each extracted frame. For example, if `frames_skip` is set to 2, every other
        frame will be extracted. If `frames_skip` is not specified, all frames will be extracted
        :type frames_skip: int
        :return: the results of calling the `parallel_map` function with the `extract_custom_frames` function from
        the `t4_utils` module, passing in the `movie_files` list as the input and the `args` tuple
        containing `output_dir`, `num_frames`, and `frames_skip`. The `parallel_map` function is a custom
        function that applies the given function to each element of a list of movie_files.
        """
        frame_modification = t_utils.clip_modification_widget()
        species_list = self.choose_species()

        button = widgets.Button(
            description="Click to modify frames",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
        )

        def on_button_clicked(b):
            movie_files = sorted(
                [
                    f
                    for f in glob.glob(f"{input_path}/*")
                    if os.path.isfile(f)
                    and os.path.splitext(f)[1].lower()
                    in [".mov", ".mp4", ".avi", ".mkv"]
                ]
            )
            results = parallel_map(
                self.extract_custom_frames,
                movie_files,
                args=(
                    [output_path] * len(movie_files),
                    [num_frames] * len(movie_files),
                    [frames_skip] * len(movie_files),
                ),
            )
            self.frames_to_upload_df = pd.concat(results)
            self.project.output_path = output_path
            self.generated_frames = self.modify_frames(
                frames_to_upload_df=self.frames_to_upload_df.reset_index(drop=True),
                species_i=species_list.value,
                modification_details=frame_modification.checks,
            )

        button.on_click(on_button_clicked)
        display(frame_modification)
        display(button)

    def get_frames(self, n_frames_subject: int = 3, subsample_up_to: int = 3):
        """
        > This function allows you to choose a species of interest, and then it will fetch a random
        sample of frames from the database for that species

        :param n_frames_subject: number of frames to fetch per subject, defaults to 3
        :type n_frames_subject: int (optional)
        :param subsample_up_to: If you have a lot of frames for a given species, you can subsample them.
               This parameter controls how many frames you want to subsample to, defaults to 3
        :type subsample_up_to: int (optional)
        """

        species_list = self.choose_species()

        button = widgets.Button(
            description="Click to fetch frames",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
        )

        def on_button_clicked(b):
            self.species_of_interest = species_list.value
            self.frames_to_upload_df = self.get_frames(
                species_names=species_list.value,
                db_path=self.db_info["db_path"],
                n_frames_subject=n_frames_subject,
                subsample_up_to=subsample_up_to,
            )

        button.on_click(on_button_clicked)
        display(button)

    # Function modify the frames
    def modify_frames(
        self,
        frames_to_upload_df: pd.DataFrame,
        species_i: list,
        modification_details: dict,
    ):
        server = self.project.server

        if len(species_i) == 0:
            species_i = ["custom_species"]

        # Specify the folder to host the modified frames
        if server == "SNIC":
            # Specify volume allocated by SNIC
            snic_path = "/mimer/NOBACKUP/groups/snic2021-6-9"
            folder_name = f"{snic_path}/tmp_dir/frames/"
            mod_frames_folder = Path(
                folder_name, "modified_" + "_".join(species_i) + "_frames/"
            )
        else:
            mod_frames_folder = "modified_" + "_".join(species_i) + "_frames/"
            if self.project.output_path is not None:
                mod_frames_folder = self.project.output_path + mod_frames_folder

        # Specify the path of the modified frames
        frames_to_upload_df["modif_frame_path"] = (
            mod_frames_folder
            + "_modified_"
            + frames_to_upload_df["frame_path"].apply(lambda x: os.path.basename(x))
        )

        # Remove existing modified clips
        if os.path.exists(mod_frames_folder):
            shutil.rmtree(mod_frames_folder)

        if len(modification_details.values()) > 0:
            # Save the modification details to include as subject metadata
            frames_to_upload_df["frame_modification_details"] = str(
                modification_details
            )

            # Create the folder to store the videos if not exist
            if not os.path.exists(mod_frames_folder):
                Path(mod_frames_folder).mkdir(parents=True, exist_ok=True)
                # Recursively add permissions to folders created
                [
                    os.chmod(root, 0o777)
                    for root, dirs, files in os.walk(mod_frames_folder)
                ]

            #### Modify the clips###
            # Read each clip and modify them (printing a progress bar)
            for index, row in tqdm(
                frames_to_upload_df.iterrows(), total=frames_to_upload_df.shape[0]
            ):
                if not os.path.exists(row["modif_frame_path"]):
                    # Set up input prompt
                    init_prompt = f"ffmpeg.input('{row['frame_path']}')"
                    full_prompt = init_prompt
                    # Set up modification
                    for transform in modification_details.values():
                        if "filter" in transform:
                            mod_prompt = transform["filter"]
                            full_prompt += mod_prompt
                    # Setup output prompt
                    crf_value = [
                        transform["crf"] if "crf" in transform else None
                        for transform in modification_details.values()
                    ]
                    crf_value = [i for i in crf_value if i is not None]

                    if len(crf_value) > 0:
                        # Note: now using q option as crf not supported by ffmpeg build
                        crf_prompt = str(max([int(i) for i in crf_value]))
                        full_prompt += f".output('{row['modif_frame_path']}', q={crf_prompt}, pix_fmt='yuv420p')"
                    else:
                        full_prompt += f".output('{row['modif_frame_path']}', q=20, pix_fmt='yuv420p')"
                    # Run the modification
                    try:
                        print(full_prompt)
                        eval(full_prompt).run(capture_stdout=True, capture_stderr=True)
                        os.chmod(row["modif_frame_path"], 0o777)
                    except ffmpeg.Error as e:
                        logging.info("stdout:", e.stdout.decode("utf8"))
                        logging.info("stderr:", e.stderr.decode("utf8"))
                        raise e

            logging.info("Frames modified successfully")

        else:
            # Save the modification details to include as subject metadata
            frames_to_upload_df["modif_frame_path"] = frames_to_upload_df["frame_path"]

        return frames_to_upload_df

    def check_frames_uploaded(self):
        """
        This function checks if the frames in the frames_to_upload_df dataframe have been uploaded to
        the database
        """
        self.check_frames_uploaded(
            self.frames_to_upload_df,
            self.project,
            self.species_of_interest,
            self.db_connection,
        )

    # t5, t6, t7
    def get_team_name(self):
        """
        > If the project name is "Spyfish_Aotearoa", return "wildlife-ai", otherwise return "koster"

        :param project_name: The name of the project you want to get the data from
        :type project_name: str
        :return: The team name is being returned.
        """

        if self.project.Project_name == "Spyfish_Aotearoa":
            return "wildlife-ai"
        else:
            return "koster"

    def choose_classes(db_path: str = "koster_lab.db"):
        """
        It creates a dropdown menu of all the species in the database, and returns the species that you
        select

        :param db_path: The path to the database, defaults to koster_lab.db
        :type db_path: str (optional)
        :return: A widget object
        """
        conn = db_utils.create_connection(db_path)
        species_list = pd.read_sql_query("SELECT label from species", conn)[
            "label"
        ].tolist()
        w = widgets.SelectMultiple(
            options=species_list,
            value=[species_list[0]],
            description="Species",
            disabled=False,
        )

        display(w)
        return w

    def get_ml_data(self):
        # get template ml data
        pass

    def process_image(self):
        # code for processing image goes here
        pass

    def prepare_metadata(self):
        # code for preparing metadata goes here
        pass

    def prepare_movies(self):
        # code for preparing movie files (standardising formats)
        pass

    # t8
    def format_to_gbif_occurence(
        self,
        df: pd.DataFrame,
        classified_by: str,
        subject_type: str,
    ):
        """
        > This function takes a df of biological observations classified by citizen scientists, biologists or ML algorithms and returns a df of species occurrences to publish in GBIF/OBIS.
        :param df: the dataframe containing the aggregated classifications
        :param classified_by: the entity who classified the object of interest, either "citizen_scientists", "biologists" or "ml_algorithms"
        :param subject_type: str,
        :param db_info_dict: a dictionary containing the path to the database and the database name
        :param project: the project object
        :param zoo_info_dict: dictionary with the workflow/subjects/classifications retrieved from Zooniverse project
        :return: a df of species occurrences to publish in GBIF/OBIS.
        """

        # If classifications have been created by citizen scientists
        if classified_by == "citizen_scientists":
            #### Retrieve subject information #####
            # Create connection to db
            conn = db_utils.create_connection(self.db_info["db_path"])

            # Add annotations to db
            zu_utils.populate_agg_annotations(df, subject_type, self.project)

            # Retrieve list of subjects
            subjects_df = pd.read_sql_query(
                "SELECT id, clip_start_time, frame_number, movie_id FROM subjects",
                conn,
            )

            # Ensure subject_ids format is int
            df["subject_ids"] = df["subject_ids"].astype(int)
            subjects_df["id"] = subjects_df["id"].astype(int)

            # Combine the aggregated clips and subjects dataframes
            comb_df = pd.merge(
                df, subjects_df, how="left", left_on="subject_ids", right_on="id"
            ).drop(columns=["id"])

            #### Retrieve movie and site information #####
            # Query info about the movie of interest
            movies_df = pd.read_sql_query("SELECT * FROM movies", conn)

            # Add survey information as part of the movie info if spyfish
            if "local_surveys_csv" in self.db_info.keys():
                # Read info about the movies
                movies_csv = pd.read_csv(self.db_info["local_movies_csv"])

                # Select only movie ids and survey ids
                movies_csv = movies_csv[["movie_id", "SurveyID"]]

                # Combine the movie_id and survey information
                movies_df = pd.merge(
                    movies_df, movies_csv, how="left", left_on="id", right_on="movie_id"
                ).drop(columns=["movie_id"])

                # Read info about the surveys
                surveys_df = pd.read_csv(
                    self.db_info["local_surveys_csv"], parse_dates=["SurveyStartDate"]
                )

                # Combine the movie_id and survey information
                movies_df = pd.merge(
                    movies_df,
                    surveys_df,
                    how="left",
                    left_on="SurveyID",
                    right_on="SurveyID",
                )

            # Combine the aggregated clips and subjects dataframes
            comb_df = pd.merge(
                comb_df, movies_df, how="left", left_on="movie_id", right_on="id"
            ).drop(columns=["id"])

            # Query info about the sites of interest
            sites_df = pd.read_sql_query("SELECT * FROM sites", conn)

            # Combine the aggregated classifications and site information
            comb_df = pd.merge(
                comb_df, sites_df, how="left", left_on="site_id", right_on="id"
            ).drop(columns=["id"])

            #### Retrieve species/labels information #####
            # Create a df with unique workflow ids and versions of interest
            work_df = (
                df[["workflow_id", "workflow_version"]].drop_duplicates().astype("int")
            )

            # Correct for some weird zooniverse version behaviour
            work_df["workflow_version"] = work_df["workflow_version"] - 1

            # Store df of all the common names and the labels into a list of df
            commonName_labels_list = [
                t_utils.get_workflow_labels(self.zoo_info["workflows"], x, y)
                for x, y in zip(work_df["workflow_id"], work_df["workflow_version"])
            ]

            # Concatenate the dfs and select only unique common names and the labels
            commonName_labels_df = pd.concat(commonName_labels_list).drop_duplicates()

            # Rename the columns as they are the other way aorund (potentially only in Spyfish?)
            vernacularName_labels_df = commonName_labels_df.rename(
                columns={
                    "commonName": "label",
                    "label": "vernacularName",
                }
            )

            # Combine the labels with the commonNames of the classifications
            comb_df = pd.merge(
                comb_df, vernacularName_labels_df, how="left", on="label"
            )

            # Query info about the species of interest
            species_df = pd.read_sql_query("SELECT * FROM species", conn)

            # Rename the column to match Darwin core std
            species_df = species_df.rename(
                columns={
                    "label": "vernacularName",
                }
            )
            # Combine the aggregated classifications and species information
            comb_df = pd.merge(comb_df, species_df, how="left", on="vernacularName")

            #### Tidy up classifications information #####
            if subject_type == "clip":
                # Identify the second of the original movie when the species first appears
                comb_df["second_in_movie"] = (
                    comb_df["clip_start_time"] + comb_df["first_seen"]
                )

            if subject_type == "frame":
                # Identify the second of the original movie when the species appears
                comb_df["second_in_movie"] = comb_df["frame_number"] * comb_df["fps"]

            # Drop the clips classified as nothing here
            comb_df = comb_df[comb_df["label"] != "NOTHINGHERE"]
            comb_df = comb_df[comb_df["label"] != "OTHER"]

            # Select the max count of each species on each movie
            comb_df = comb_df.sort_values("how_many").drop_duplicates(
                ["movie_id", "vernacularName"], keep="last"
            )

            # Rename columns to match Darwin Data Core Standards
            comb_df = comb_df.rename(
                columns={
                    "created_on": "eventDate",
                    "how_many": "individualCount",
                }
            )

            # Create relevant columns for GBIF
            comb_df["occurrenceID"] = (
                self.project.Project_name
                + "_"
                + comb_df["siteName"]
                + "_"
                + comb_df["eventDate"].astype(str)
                + "_"
                + comb_df["second_in_movie"].astype(str)
                + "_"
                + comb_df["vernacularName"].astype(str)
            )

            comb_df["basisOfRecord"] = "MachineObservation"

            # If coord uncertainity doesn't exist set to 30 metres
            comb_df["coordinateUncertaintyInMeters"] = comb_df.get(
                "coordinateUncertaintyInMeters", 30
            )

            # Select columns relevant for GBIF occurrences
            comb_df = comb_df[
                [
                    "occurrenceID",
                    "basisOfRecord",
                    "vernacularName",
                    "scientificName",
                    "eventDate",
                    "countryCode",
                    "taxonRank",
                    "kingdom",
                    "decimalLatitude",
                    "decimalLongitude",
                    "geodeticDatum",
                    "coordinateUncertaintyInMeters",
                    "individualCount",
                ]
            ]

            return comb_df

        # If classifications have been created by biologists
        if classified_by == "biologists":
            logging.info("This sections is currently under development")

        # If classifications have been created by ml algorithms
        if classified_by == "ml_algorithms":
            logging.info("This sections is currently under development")
        else:
            raise ValueError(
                "Specify who classified the species of interest (citizen_scientists, biologists or ml_algorithms)"
            )

    def get_classifications(
        self,
        workflow_dict: dict,
        workflows_df: pd.DataFrame,
        subj_type: str,
        class_df: pd.DataFrame,
        db_path: str,
    ):
        """
        It takes in a dictionary of workflows, a dataframe of workflows, the type of subject (frame or
        clip), a dataframe of classifications, the path to the database, and the project name. It returns a
        dataframe of classifications

        :param workflow_dict: a dictionary of the workflows you want to retrieve classifications for. The
            keys are the workflow names, and the values are the workflow IDs, workflow versions, and the minimum
            number of classifications per subject
        :type workflow_dict: dict
        :param workflows_df: the dataframe of workflows from the Zooniverse project
        :type workflows_df: pd.DataFrame
        :param subj_type: "frame" or "clip"
        :param class_df: the dataframe of classifications from the database
        :param db_path: the path to the database file
        :param project: the name of the project on Zooniverse
        :return: A dataframe with the classifications for the specified project and workflow.
        """

        names, workflow_versions = [], []
        for i in range(0, len(workflow_dict), 3):
            names.append(list(workflow_dict.values())[i])
            workflow_versions.append(list(workflow_dict.values())[i + 2])

        workflow_ids = t_utils.get_workflow_ids(workflows_df, names)

        # Filter classifications of interest
        classes = []
        for id, version in zip(workflow_ids, workflow_versions):
            class_df_id = class_df[
                (class_df.workflow_id == id) & (class_df.workflow_version >= version)
            ].reset_index(drop=True)
            classes.append(class_df_id)
        classes_df = pd.concat(classes)

        # Add information about the subject
        # Create connection to db
        conn = db_utils.create_connection(db_path)

        if subj_type == "frame":
            # Query id and subject type from the subjects table
            subjects_df = pd.read_sql_query(
                "SELECT id, subject_type, \
                                            https_location, filename, frame_number, movie_id FROM subjects \
                                            WHERE subject_type=='frame'",
                conn,
            )

        else:
            # Query id and subject type from the subjects table
            subjects_df = pd.read_sql_query(
                "SELECT id, subject_type, \
                                            https_location, filename, clip_start_time, movie_id FROM subjects \
                                            WHERE subject_type=='clip'",
                conn,
            )

        # Ensure id format matches classification's subject_id
        classes_df["subject_ids"] = classes_df["subject_ids"].astype("Int64")
        subjects_df["id"] = subjects_df["id"].astype("Int64")

        # Add subject information based on subject_ids
        classes_df = pd.merge(
            classes_df,
            subjects_df,
            how="left",
            left_on="subject_ids",
            right_on="id",
        )

        if classes_df[["subject_type", "https_location"]].isna().any().any():
            # Exclude classifications from missing subjects
            filtered_class_df = classes_df.dropna(
                subset=["subject_type", "https_location"], how="any"
            ).reset_index(drop=True)

            # Report on the issue
            logging.info(
                f"There are {(classes_df.shape[0]-filtered_class_df.shape[0])}"
                f" classifications out of {classes_df.shape[0]}"
                f" missing subject info. Maybe the subjects have been removed from Zooniverse?"
            )

            classes_df = filtered_class_df

        logging.info(
            f"{classes_df.shape[0]} Zooniverse classifications have been retrieved"
        )

        return classes_df

    def process_classifications(
        self,
        classifications_data,
        subject_type: str,
        agg_params: list,
        summary: bool = False,
    ):
        """
        It takes in a dataframe of classifications, a subject type (clip or frame), a list of
        aggregation parameters, and a boolean for whether or not to return a summary of the
        classifications.

        It then returns a dataframe of aggregated classifications.

        Let's break it down.

        First, we check that the length of the aggregation parameters is correct for the subject type.

        Then, we define a function called `get_classifications` that takes in a dataframe of
        classifications and a subject type.

        This function queries the subjects table for the subject type and then merges the
        classifications dataframe with the subjects dataframe.

        It then returns the merged dataframe.

        Finally, we call the `aggregrate_classifications` function, passing
        in the dataframe returned by `get_classifications`, the subject

        :param classifications_data: the dataframe of classifications from the Zooniverse API
        :param subject_type: This is the type of subject you want to retrieve classifications for. This
               can be either "clip" or "frame"
        :type subject_type: str
        :param agg_params: list
        :type agg_params: list
        :param summary: If True, the output will be a summary of the classifications, with the number of
               classifications per label, defaults to False
        :type summary: bool (optional)
        """

        t = False
        if subject_type == "clip":
            t = len(agg_params) == 2
        elif subject_type == "frame":
            t = len(agg_params) == 5

        if not t:
            logging.error("Incorrect agg_params length for subject type")
            return

        def get_classifications(classes_df, subject_type):
            conn = self.db_connection
            if subject_type == "frame":
                # Query id and subject type from the subjects table
                subjects_df = pd.read_sql_query(
                    "SELECT id, subject_type, \
                                                https_location, filename, frame_number, movie_id FROM subjects \
                                                WHERE subject_type=='frame'",
                    conn,
                )

            else:
                # Query id and subject type from the subjects table
                subjects_df = pd.read_sql_query(
                    "SELECT id, subject_type, \
                                                https_location, filename, clip_start_time, movie_id FROM subjects \
                                                WHERE subject_type=='clip'",
                    conn,
                )

            # Ensure id format matches classification's subject_id
            classes_df["subject_ids"] = classes_df["subject_ids"].astype("Int64")
            subjects_df["id"] = subjects_df["id"].astype("Int64")

            # Add subject information based on subject_ids
            classes_df = pd.merge(
                classes_df,
                subjects_df,
                how="left",
                left_on="subject_ids",
                right_on="id",
            )

            if classes_df[["subject_type", "https_location"]].isna().any().any():
                # Exclude classifications from missing subjects
                filtered_class_df = classes_df.dropna(
                    subset=["subject_type", "https_location"], how="any"
                ).reset_index(drop=True)

                # Report on the issue
                logging.info(
                    f"There are {(classes_df.shape[0]-filtered_class_df.shape[0])}"
                    f" classifications out of {classes_df.shape[0]}"
                    f" missing subject info. Maybe the subjects have been removed from Zooniverse?"
                )

                classes_df = filtered_class_df

            logging.info(
                f"{classes_df.shape[0]} Zooniverse classifications have been retrieved"
            )

            return classes_df

        agg_class_df, raw_class_df = self.aggregate_classifications(
            get_classifications(classifications_data, subject_type),
            subject_type,
            agg_params,
        )
        if summary:
            agg_class_df = (
                agg_class_df.groupby("label")["subject_ids"].agg("count").to_frame()
            )
        return agg_class_df, raw_class_df

    def process_annotations(self):
        # code for prepare dataset for machine learning
        pass

    def format_to_gbif(self, agg_df: pd.DataFrame, subject_type: str):
        return self.format_to_gbif_occurence(
            df=agg_df,
            classified_by="citizen_scientists",
            subject_type=subject_type,
        )


class MLProjectProcessor(ProjectProcessor):
    def __init__(
        self,
        project_process: ProjectProcessor,
        config_path: str = None,
        weights_path: str = None,
        output_path: str = None,
        classes: list = [],
    ):
        self.__dict__ = project_process.__dict__.copy()
        self.project_name = self.project.Project_name.lower().replace(" ", "_")
        self.data_path = config_path
        self.weights_path = weights_path
        self.output_path = output_path
        self.classes = classes
        self.run_history = None
        self.best_model_path = None
        self.model_type = None
        self.train, self.run, self.test = (None,) * 3
        self.modules = import_modules([])
        self.modules.update(
            import_modules(["torch", "wandb", "yaml", "yolov5"], utils=False)
        )

        self.team_name = self.get_team_name()

        model_selected = t_utils.choose_model_type()

        async def f():
            x = await t_utils.single_wait_for_change(model_selected, "value")
            self.model_type = x
            self.modules.update(self.load_yolov5_modules())
            if all(["train", "detect", "val"]) in self.modules:
                self.train, self.run, self.test = (
                    self.modules["train"],
                    self.modules["detect"],
                    self.modules["val"],
                )

        asyncio.create_task(f())

    def load_yolov5_modules(self):
        # Model-specific imports
        if self.model_type == 1:
            module_names = ["yolov5.train", "yolov5.detect", "yolov5.val"]
            logging.info("Object detection model loaded")
            return import_modules(module_names, utils=False, models=True)
        elif self.model_type == 2:
            logging.info("Image classification model loaded")
            module_names = [
                "yolov5.classify.train",
                "yolov5.classify.predict",
                "yolov5.classify.val",
            ]
            return import_modules(module_names, utils=False, models=True)
        elif self.model_type == 3:
            logging.info("Image segmentation model loaded")
            module_names = [
                "yolov5.segment.train",
                "yolov5.segment.predict",
                "yolov5.segment.val",
            ]
            return import_modules(module_names, utils=False, models=True)
        else:
            logging.info("Invalid model specification")

    def prepare_dataset(
        self,
        agg_df: pd.DataFrame,
        out_path: str,
        perc_test: float = 0.2,
        img_size: tuple = (224, 224),
        remove_nulls: bool = False,
        track_frames: bool = False,
        n_tracked_frames: int = 0,
    ):
        species_list = self.choose_species()

        button = widgets.Button(
            description="Aggregate frames",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
            style={"description_width": "initial"},
        )

        def on_button_clicked(b):
            self.species_of_interest = species_list.value
            # code for prepare dataset for machine learning
            yolo_utils.frame_aggregation(
                self.project,
                self.db_info,
                out_path,
                perc_test,
                self.species_of_interest,
                img_size,
                remove_nulls,
                track_frames,
                n_tracked_frames,
                agg_df,
            )

        button.on_click(on_button_clicked)
        display(button)

    def choose_entity(self, alt_name: bool = False):
        if self.team_name is None:
            return t_utils.choose_entity()
        else:
            if not alt_name:
                logging.info(
                    f"Found team name: {self.team_name}. If you want"
                    " to use a different team name for this experiment"
                    " set the argument alt_name to True"
                )
            else:
                return t_utils.choose_entity()

    # Function to choose a model to evaluate
    def choose_model(self):
        """
        It takes a project name and returns a dropdown widget that displays the metrics of the model
        selected

        :param project_name: The name of the project you want to load the model from
        :return: The model_widget is being returned.
        """
        model_dict = {}
        model_info = {}
        api = wandb.Api()
        # weird error fix (initialize api another time)

        project_name = self.project.Project_name.replace(" ", "_")
        if self.team_name == "wildlife-ai":
            logging.info("Please note: Using models from adi-ohad-heb-uni account.")
            full_path = "adi-ohad-heb-uni/project-wildlife-ai"
            api.runs(path=full_path).objects
        else:
            full_path = f"{self.team_name}/{project_name.lower()}"

        runs = api.runs(full_path)

        for run in runs:
            model_artifacts = [
                artifact
                for artifact in chain(run.logged_artifacts(), run.used_artifacts())
                if artifact.type == "model"
            ]
            if len(model_artifacts) > 0:
                model_dict[run.name] = model_artifacts[0].name.split(":")[0]
                model_info[model_artifacts[0].name.split(":")[0]] = run.summary

        # Add "no movie" option to prevent conflicts
        # models = np.append(list(model_dict.keys()),"No model")

        model_widget = widgets.Dropdown(
            options=[(name, model) for name, model in model_dict.items()],
            description="Select model:",
            ensure_option=False,
            disabled=False,
            layout=widgets.Layout(width="50%"),
            style={"description_width": "initial"},
        )

        main_out = widgets.Output()
        display(model_widget, main_out)

        # Display model metrics
        def on_change(change):
            with main_out:
                clear_output()
                if change["new"] == "No file":
                    logging.info("Choose another file")
                else:
                    if project_name == "model-registry":
                        logging.info("No metrics available")
                    else:
                        logging.info(
                            {
                                k: v
                                for k, v in model_info[change["new"]].items()
                                if "metrics" in k
                            }
                        )

        model_widget.observe(on_change, names="value")
        return model_widget

    def transfer_model(
        model_name: str, artifact_dir: str, project_name: str, user: str, password: str
    ):
        """
        It takes the model name, the artifact directory, the project name, the user and the password as
        arguments and then downloads the latest model from the project and uploads it to the server

        :param model_name: the name of the model you want to transfer
        :type model_name: str
        :param artifact_dir: the directory where the model is stored
        :type artifact_dir: str
        :param project_name: The name of the project you want to transfer the model from
        :type project_name: str
        :param user: the username of the remote server
        :type user: str
        :param password: the password for the user you're using to connect to the server
        :type password: str
        """
        ssh = SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.load_system_host_keys()
        ssh.connect(
            hostname="80.252.221.46", port=2230, username=user, password=password
        )

        # SCPCLient takes a paramiko transport as its only argument
        scp = SCPClient(ssh.get_transport())
        scp.put(
            f"{artifact_dir}/weights/best.pt",
            f"/home/koster/model_config/weights/ \
                {os.path.basename(project_name)}_{os.path.basename(os.path.dirname(artifact_dir))}_{model_name}",
        )
        scp.close()

    def setup_paths(self):
        if not isinstance(self.output_path, str) and self.output_path is not None:
            self.output_path = self.output_path.selected
        self.data_path, self.hyp_path = yolo_utils.setup_paths(
            self.output_path, self.model_type
        )

    def choose_train_params(self):
        return t_utils.choose_train_params(self.model_type)

    def train_yolov5(
        self, exp_name, weights, epochs=50, batch_size=16, img_size=[720, 540]
    ):
        if self.model_type == 1:
            self.modules["train"].run(
                entity=self.team_name,
                data=self.data_path,
                hyp=self.hyp_path,
                weights=weights,
                project=self.project_name,
                name=exp_name,
                img_size=img_size,
                batch_size=int(batch_size),
                epochs=epochs,
                workers=1,
                single_cls=False,
                cache_images=True,
            )
        elif self.model_type == 2:
            self.modules["train"].run(
                entity=self.team_name,
                data=self.data_path,
                model=weights,
                project=self.project_name,
                name=exp_name,
                img_size=img_size[0],
                batch_size=int(batch_size),
                epochs=epochs,
                workers=1,
            )
        else:
            logging.error("Segmentation model training not yet supported.")

    def eval_yolov5(self, exp_name: str, model_folder: str, conf_thres: float):
        # Find trained model weights
        project_path = str(Path(self.output_path, self.project.Project_name.lower()))
        self.tuned_weights = f"{Path(project_path, model_folder, 'weights', 'best.pt')}"
        try:
            self.modules["val"].run(
                data=self.data_path,
                weights=self.tuned_weights,
                conf_thres=conf_thres,
                imgsz=640 if self.model_type == 1 else 224,
                half=False,
                project=self.project_name,
                name=str(exp_name) + "_val",
            )
        except Exception as e:
            logging.error(f"Encountered {e}, terminating run...")
            self.modules["wandb"].finish()
        logging.info("Run succeeded, finishing run...")
        self.modules["wandb"].finish()

    def detect_yolov5(
        self, source: str, save_dir: str, conf_thres: float, artifact_dir: str
    ):
        self.run = self.modules["wandb"].init(
            entity=self.team_name,
            project="model-evaluations",
            settings=self.modules["wandb"].Settings(start_method="fork"),
        )
        self.modules["detect"].run(
            weights=[
                f
                for f in Path(artifact_dir).iterdir()
                if f.is_file()
                and str(f).endswith((".pt", ".model"))
                and "osnet" not in str(f)
            ][0],
            source=source,
            conf_thres=conf_thres,
            save_txt=True,
            save_conf=True,
            project=save_dir,
            name="detect",
        )

    def save_detections_wandb(self, conf_thres: float, model: str, eval_dir: str):
        yolo_utils.set_config(conf_thres, model, eval_dir)
        yolo_utils.add_data_wandb(eval_dir, "detection_output", self.run)
        self.csv_report = yolo_utils.generate_csv_report(
            eval_dir.selected, wandb_log=True
        )
        wandb.finish()

    def track_individuals(
        self,
        source: str,
        artifact_dir: str,
        eval_dir: str,
        conf_thres: float,
        img_size: tuple = (540, 540),
    ):
        latest_tracker = yolo_utils.track_objects(
            source_dir=source,
            artifact_dir=artifact_dir,
            tracker_folder=eval_dir,
            conf_thres=conf_thres,
            img_size=img_size,
            gpu=True if self.modules["torch"].cuda.is_available() else False,
        )
        yolo_utils.add_data_wandb(
            Path(latest_tracker).parent.absolute(), "tracker_output", self.run
        )
        self.csv_report = yolo_utils.generate_csv_report(eval_dir, wandb_log=True)
        self.tracking_report = yolo_utils.generate_counts(
            eval_dir, latest_tracker, artifact_dir, wandb_log=True
        )
        self.modules["wandb"].finish()

    def enhance_yolov5(self, conf_thres: float, img_size=[640, 640]):
        if self.model_type == 1:
            logging.info("Enhancement running...")
            self.modules["detect"].run(
                weights=self.tuned_weights,
                source=str(Path(self.output_path, "images")),
                imgsz=img_size,
                conf_thres=conf_thres,
                save_txt=True,
            )
            self.modules["wandb"].finish()
        elif self.model_type == 2:
            logging.info(
                "Enhancements not supported for image classification models at this time."
            )
        else:
            logging.info(
                "Enhancements not supported for segmentation models at this time."
            )

    def enhance_replace(self, run_folder: str):
        if self.model_type == 1:
            os.move(f"{self.output_path}/labels", f"{self.output_path}/labels_org")
            os.move(f"{run_folder}/labels", f"{self.output_path}/labels")
        else:
            logging.error("This option is not supported for other model types.")

    def download_project_runs(self):
        # Download all the runs from the given project ID using Weights and Biases API,
        # sort them by the specified metric, and assign them to the run_history attribute

        self.modules["wandb"].login()
        runs = self.modules["wandb"].Api().runs(f"{self.team_name}/{self.project_name}")
        self.run_history = []
        for run in runs:
            run_info = {}
            run_info["run"] = run
            metrics = run.history()
            run_info["metrics"] = metrics
            self.run_history.append(run_info)
        # self.run_history = sorted(
        #    self.run_history, key=lambda x: x["metrics"]["metrics/"+sort_metric]
        # )

    def get_model(self, model_name: str, download_path: str):
        """
        It downloads the latest model checkpoint from the specified project and model name

        :param model_name: The name of the model you want to download
        :type model_name: str
        :param project_name: The name of the project you want to download the model from
        :type project_name: str
        :param download_path: The path to download the model to
        :type download_path: str
        :return: The path to the downloaded model checkpoint.
        """
        if self.team_name == "wildlife-ai":
            logging.info("Please note: Using models from adi-ohad-heb-uni account.")
            full_path = "adi-ohad-heb-uni/project-wildlife-ai"
        else:
            full_path = f"{self.team_name}/{self.project.Project_name.lower()}"
        api = wandb.Api()
        try:
            api.artifact_type(type_name="model", project=full_path).collections()
        except Exception as e:
            logging.error(
                f"No model collections found. No artifacts have been logged. {e}"
            )
            return None
        collections = [
            coll
            for coll in api.artifact_type(
                type_name="model", project=full_path
            ).collections()
        ]
        model = [i for i in collections if i.name == model_name]
        if len(model) > 0:
            model = model[0]
        else:
            logging.error("No model found")
        artifact = api.artifact(full_path + "/" + model.name + ":latest")
        logging.info("Downloading model checkpoint...")
        artifact_dir = artifact.download(root=download_path)
        logging.info("Checkpoint downloaded.")
        return os.path.realpath(artifact_dir)

    def get_best_model(self, metric="mAP_0.5", download_path: str = ""):
        # Get the best model from the run history according to the specified metric
        if self.run_history is not None:
            best_run = self.run_history[0]
        else:
            self.download_project_runs()
            best_run = self.run_history[0]
        try:
            best_metric = best_run["metrics"][metric]
            for run in self.run_history:
                if run["metrics"][metric] < best_metric:
                    best_run = run
                    best_metric = run["metrics"][metric]
        except KeyError:
            logging.error(
                "No run with the given metric has been recorded. Using first run as best run."
            )
        best_model = [
            artifact
            for artifact in chain(
                best_run["run"].logged_artifacts(), best_run["run"].used_artifacts()
            )
            if artifact.type == "model"
        ][0]

        api = self.modules["wandb"].Api()
        artifact = api.artifact(
            f"{self.team_name}/{self.project_name}"
            + "/"
            + best_model.name.split(":")[0]
            + ":latest"
        )
        logging.info("Downloading model checkpoint...")
        artifact_dir = artifact.download(root=download_path)
        logging.info("Checkpoint downloaded.")
        self.best_model_path = os.path.realpath(artifact_dir)

    def export_best_model(self, output_path):
        # Export the best model to PyTorch format
        import torch
        import tensorflow as tf

        model = tf.keras.models.load_model(self.best_model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open("temp.tflite", "wb") as f:
            f.write(tflite_model)
        converter = torch.onnx.TFLiteParser.parse("temp.tflite")
        with open(output_path, "wb") as f:
            f.write(converter)

    def get_dataset(self, model: str, team_name: str = "koster"):
        """
        It takes in a project name and a model name, and returns the paths to the train and val datasets

        :param project_name: The name of the project you want to download the dataset from
        :type project_name: str
        :param model: The model you want to use
        :type model: str
        :return: The return value is a list of two directories, one for the training data and one for the validation data.
        """
        api = wandb.Api()
        if "_" in model:
            run_id = model.split("_")[1]
            try:
                run = api.run(
                    f"{team_name}/{self.project.Project_name.lower()}/runs/{run_id}"
                )
            except wandb.CommError:
                logging.error("Run data not found")
                return "empty_string", "empty_string"
            datasets = [
                artifact
                for artifact in run.used_artifacts()
                if artifact.type == "dataset"
            ]
            if len(datasets) == 0:
                logging.error(
                    "No datasets are linked to these runs. Please try another run."
                )
                return "empty_string", "empty_string"
            dirs = []
            for i in range(len(["train", "val"])):
                artifact = datasets[i]
                logging.info(f"Downloading {artifact.name} checkpoint...")
                artifact_dir = artifact.download()
                logging.info(f"{artifact.name} - Dataset downloaded.")
                dirs.append(artifact_dir)
            return dirs
        else:
            logging.error("Externally trained model. No data available.")
            return "empty_string", "empty_string"


class Annotator:
    def __init__(self, dataset_name, images_path, potential_labels=None):
        self.dataset_name = dataset_name
        self.images_path = images_path
        self.potential_labels = potential_labels
        self.bboxes = {}
        self.modules = import_modules(["t5_utils", "t6_utils", "t7_utils"])
        self.modules.update(import_modules(["fiftyone"], utils=False))

    def __repr__(self):
        return repr(self.__dict__)

    def fiftyone_annotate(self):
        # Create a new dataset
        try:
            dataset = self.modules["fiftyone"].load_dataset(self.dataset_name)
            dataset.delete()
        except ValueError:
            pass
        dataset = self.modules["fiftyone"].Dataset(self.dataset_name)

        # Add all the images in the directory to the dataset
        for filename in os.listdir(self.images_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(self.images_path, filename)
                sample = self.modules["fiftyone"].Sample(filepath=image_path)
                dataset.add_sample(sample)

        # Add the potential labels to the dataset
        # Set default classes
        if self.potential_labels is not None:
            label_field = "my_label"
            dataset.add_sample_field(
                label_field,
                self.modules["fiftyone"].core.fields.StringField,
                classes=self.potential_labels,
            )

        # Create a view with the desired labels

        dataset.annotate(
            self.dataset_name,
            label_type="scalar",
            label_field=label_field,
            launch_editor=True,
            backend="labelbox",
        )
        # Open the dataset in the FiftyOne App
        # Connect to FiftyOne session
        # session = self.modules["fiftyone"].launch_app(dataset, view=view)

        # Start annotating
        # session.wait()

        # Save the annotations
        dataset.save()

    def annotate(self, autolabel_model: str = None):
        return t_utils.get_annotator(
            self.images_path, self.potential_labels, autolabel_model
        )

    def load_annotations(self):
        images = sorted(
            [
                f
                for f in os.listdir(self.images_path)
                if os.path.isfile(os.path.join(self.images_path, f))
                and f.endswith(".jpg")
            ]
        )
        bbox_dict = {}
        annot_path = os.path.join(Path(self.images_path).parent, "labels")
        if len(os.listdir(annot_path)) > 0:
            for label_file in os.listdir(annot_path):
                image = os.path.join(self.images_path, images[0])
                width, height = imagesize.get(image)
                bboxes = []
                bbox_dict[image] = []
                with open(os.path.join(annot_path, label_file), "r") as f:
                    for line in f:
                        s = line.split(" ")
                        left = (float(s[1]) - (float(s[3]) / 2)) * width
                        top = (float(s[2]) - (float(s[4]) / 2)) * height
                        bbox_dict[image].append(
                            {
                                "x": left,
                                "y": top,
                                "width": float(s[3]) * width,
                                "height": float(s[4]) * height,
                                "label": self.potential_labels[int(s[0])],
                            }
                        )
                        bboxes.append(
                            {
                                "x": left,
                                "y": top,
                                "width": float(s[3]) * width,
                                "height": float(s[4]) * height,
                                "label": self.potential_labels[int(s[0])],
                            }
                        )
            self.bboxes = bbox_dict
        else:
            self.bboxes = {}
