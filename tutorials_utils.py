# base imports
import os
import pandas as pd
import logging
import subprocess
from urllib.parse import urlparse
from urllib.request import pathname2url

# widget imports
import ipywidgets as widgets
from ipyfilechooser import FileChooser
from IPython.display import HTML, display
from ipywidgets import interactive
import asyncio

from panoptes_client import Project

# util imports
import kso_utils.server_utils as server_utils
import kso_utils.db_utils as db_utils
import kso_utils.zooniverse_utils as zooniverse_utils
import kso_utils.project_utils as project_utils
import kso_utils.movie_utils as movie_utils


# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def choose_folder(start_path: str = ".", folder_type: str = ""):
    # Specify the output folder
    fc = FileChooser(start_path)
    fc.title = f"Choose location of {folder_type}"
    display(fc)
    return fc


def get_project_info(projects_csv: str, project_name: str, info_interest: str):
    """
    > This function takes in a csv file of project information, a project name, and a column of interest
    from the csv file, and returns the value of the column of interest for the project name

    :param projects_csv: the path to the csv file containing the list of projects
    :param project_name: The name of the project you want to get the info for
    :param info_interest: the column name of the information you want to get from the project info
    :return: The project info
    """

    # Read the latest list of projects
    projects_df = pd.read_csv(projects_csv)

    # Get the info_interest from the project info
    project_info = projects_df[projects_df["Project_name"] == project_name][
        info_interest
    ].unique()[0]

    return project_info


def choose_project(projects_csv: str = "../db_starter/projects_list.csv"):
    """
    > This function takes a csv file with a list of projects and returns a dropdown menu with the
    projects listed

    :param projects_csv: str = "../db_starter/projects_list.csv", defaults to
    ../db_starter/projects_list.csv
    :type projects_csv: str (optional)
    :return: A dropdown widget with the project names as options.
    """

    # Check path to the list of projects is a csv
    if os.path.exists(projects_csv) and not projects_csv.endswith(".csv"):
        logging.error("A csv file was not selected. Please try again.")

    # If list of projects doesn't exist retrieve it from github
    if not os.path.exists(projects_csv):
        projects_csv = "https://github.com/ocean-data-factory-sweden/koster_data_management/blob/main/db_starter/projects_list.csv?raw=true"

    projects_df = pd.read_csv(projects_csv)

    if "Project_name" not in projects_df.columns:
        logging.error(
            "We were unable to find any projects in that file, \
                      please choose a projects csv file that matches our template."
        )

    # Display the project options
    choose_project = widgets.Dropdown(
        options=projects_df.Project_name.unique().tolist(),
        value=projects_df.Project_name.unique().tolist()[0],
        description="Project:",
        disabled=False,
    )

    display(choose_project)
    return choose_project


def get_project_details(project: project_utils.Project):
    """
    > This function connects to the server (or folder) hosting the csv files, and gets the initial info
    from the database

    :param project: the project object
    """

    # Connect to the server (or folder) hosting the csv files
    server_i_dict = server_utils.connect_to_server(project)

    # Get the initial info
    db_initial_info = server_utils.get_db_init_info(project, server_i_dict)

    return server_i_dict, db_initial_info


def initiate_db(project: project_utils.Project):
    """
    This function takes a project name as input and returns a dictionary with all the information needed
    to connect to the project's database

    :param project: The name of the project. This is used to get the project-specific info from the
    config file
    :return: A dictionary with the following keys:
        - db_path
        - project_name
        - server_i_dict
        - db_initial_info
    """

    # Get project specific info
    server_i_dict, db_initial_info = get_project_details(project)

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
            db_utils.populate_db(
                db_initial_info=db_initial_info, project=project, local_csv=local_i_csv
            )

    # Combine server/project info in a dictionary
    db_info_dict = {**db_initial_info, **server_i_dict}

    return db_info_dict


def connect_zoo_project(project: project_utils.Project):
    """
    It takes a project name as input, and returns a Zooniverse project object

    :param project: the project you want to connect to
    :return: A Zooniverse project object.
    """
    # Save your Zooniverse user name and password.
    zoo_user, zoo_pass = zooniverse_utils.zoo_credentials()

    # Get the project-specific zooniverse number
    project_n = project.Zooniverse_number

    # Connect to the Zooniverse project
    project = zooniverse_utils.auth_session(zoo_user, zoo_pass, project_n)

    logging.info("Connected to Zooniverse")

    return project


def retrieve__populate_zoo_info(
    project: project_utils.Project,
    db_info_dict: dict,
    zoo_project: Project,
    zoo_info: str,
):
    """
    It retrieves the information of the subjects uploaded to Zooniverse and populates the SQL database
    with the information

    :param project: the project you want to retrieve information for
    :param db_info_dict: a dictionary containing the path to the database and the name of the database
    :param zoo_project: The name of the Zooniverse project you created
    :param zoo_info: a string containing the information of the Zooniverse project
    :return: The zoo_info_dict is being returned.
    """

    if zoo_project is None:
        logging.error(
            "This project is not linked to a Zooniverse project. Please create one and add the required fields to proceed with this tutorial."
        )
    else:
        # Retrieve and store the information of subjects uploaded to zooniverse
        zoo_info_dict = zooniverse_utils.retrieve_zoo_info(
            project, zoo_project, zoo_info
        )

        # Populate the sql with subjects uploaded to Zooniverse
        zooniverse_utils.populate_subjects(
            zoo_info_dict["subjects"], project, db_info_dict["db_path"]
        )
        return zoo_info_dict


def choose_single_workflow(workflows_df: pd.DataFrame):
    """
    > This function displays two dropdown menus, one for the workflow name and one for the subject type

    :param workflows_df: a dataframe containing the workflows you want to choose from
    :return: the workflow name and subject type.
    """

    # Display the names of the workflows
    workflow_name = widgets.Dropdown(
        options=workflows_df.display_name.unique().tolist(),
        value=workflows_df.display_name.unique().tolist()[0],
        description="Workflow name:",
        disabled=False,
    )

    # Display the type of subjects
    subj_type = widgets.Dropdown(
        options=["frame", "clip"],
        value="clip",
        description="Subject type:",
        disabled=False,
    )

    display(workflow_name)
    display(subj_type)

    return workflow_name, subj_type


# Select the movie you want
def select_movie(available_movies_df: pd.DataFrame):
    """
    > This function takes in a dataframe of available movies and returns a widget that allows the user
    to select a movie of interest

    :param available_movies_df: a dataframe containing the list of available movies
    :return: The widget object
    """

    # Get the list of available movies
    available_movies_tuple = tuple(sorted(available_movies_df.filename.unique()))

    # Widget to select the movie
    select_movie_widget = widgets.Dropdown(
        options=available_movies_tuple,
        description="Movie of interest:",
        ensure_option=True,
        disabled=False,
        layout=widgets.Layout(width="50%"),
        style={"description_width": "initial"},
    )

    display(select_movie_widget)
    return select_movie_widget


# Function to preview underwater movies
def preview_movie(
    project: project_utils.Project,
    db_info_dict: dict,
    available_movies_df: pd.DataFrame,
    movie_i: str,
):
    """
    It takes a movie filename and returns a HTML object that can be displayed in the notebook

    :param project: the project object
    :param db_info_dict: a dictionary containing the database information
    :param available_movies_df: a dataframe with all the movies in the database
    :param movie_i: the filename of the movie you want to preview
    :return: A tuple of two elements:
        1. HTML object
        2. Movie path
    """

    # Select the movie of interest
    movie_selected = available_movies_df[
        available_movies_df["filename"] == movie_i
    ].reset_index(drop=True)
    movie_selected_view = movie_selected.T
    movie_selected_view.columns = ["Movie summary"]

    # Make sure only one movie is selected
    if len(movie_selected.index) > 1:
        logging.info(
            "There are several movies with the same filename. This should be fixed!"
        )
        return None

    else:
        # Generate temporary path to the movie select
        if project.server == "SNIC":
            movie_path = movie_utils.get_movie_path(
                project=project,
                db_info_dict=db_info_dict,
                f_path=movie_selected["spath"].values[0],
            )
            url = (
                "https://portal.c3se.chalmers.se/pun/sys/dashboard/files/fs/"
                + pathname2url(movie_path)
            )
        else:
            url = movie_utils.get_movie_path(
                f_path=movie_selected["fpath"].values[0],
                db_info_dict=db_info_dict,
                project=project,
            )
            movie_path = url
        html_code = f"""<html>
                <div style="display: flex; justify-content: space-around; align-items: center">
                <div>
                  <video width=500 controls>
                  <source src={url}>
                  </video>
                </div>
                <div>{movie_selected_view.to_html()}</div>
                </div>
                </html>"""
        return HTML(html_code), movie_path


# Function to update widget based on user interaction (eg. click)
def wait_for_change(widget1: widgets.Widget, widget2: widgets.Widget):
    future = asyncio.Future()

    def getvalue(change):
        future.set_result(change.description)
        widget1.on_click(getvalue, remove=True)
        widget2.on_click(getvalue, remove=True)

    widget1.on_click(getvalue)
    widget2.on_click(getvalue)
    return future


# Function to check if an url is valid or not
def is_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def gpu_select():
    """
    If the user selects "No GPU", then the function will return a boolean value of False. If the user
    selects "Colab GPU", then the function will install the GPU requirements and return a boolean value
    of True. If the user selects "Other GPU", then the function will return a boolean value of True
    :return: The gpu_available variable is being returned.
    """

    def gpu_output(gpu_option):
        if gpu_option == "No GPU":
            logging.info("You are set to start the modifications")
            # Set GPU argument
            gpu_available = False
            return gpu_available

        if gpu_option == "Colab GPU":
            # Install the GPU requirements
            if not os.path.exists("./colab-ffmpeg-cuda/bin/."):
                try:
                    logging.info(
                        "Installing the GPU requirements. PLEASE WAIT 10-20 SECONDS"
                    )  # Install ffmpeg with GPU version
                    subprocess.check_call(
                        "git clone https://github.com/fritolays/colab-ffmpeg-cuda.git",
                        shell=True,
                    )
                    subprocess.check_call(
                        "cp -r ./colab-ffmpeg-cuda/bin/. /usr/bin/", shell=True
                    )
                    logging.info("GPU Requirements installed!")

                except subprocess.CalledProcessError as e:
                    logging.error(
                        f"There was an issues trying to install the GPU requirements, {e}"
                    )

            # Set GPU argument
            gpu_available = True
            return gpu_available

        if gpu_option == "Other GPU":
            # Set GPU argument
            gpu_available = True
            return gpu_available

    # Select the gpu availability
    gpu_output_interact = interactive(
        gpu_output,
        gpu_option=widgets.RadioButtons(
            options=["No GPU", "Colab GPU", "Other GPU"],
            value="No GPU",
            description="Select GPU availability:",
            disabled=False,
        ),
    )

    display(gpu_output_interact)
    return gpu_output_interact
