# base imports
import os
import pandas as pd
import logging
import subprocess
from urllib.parse import urlparse

# widget imports
import ipywidgets as widgets
from ipyfilechooser import FileChooser
from IPython.display import HTML, display
from ipywidgets import interactive
import asyncio

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def process_source(source):
    """
    If the source is a string, write the string to a file and return the file name. If the source is a
    list, return the list. If the source is neither, return None

    :param source: The source of the data. This can be a URL, a file, or a list of URLs or files
    :return: the value of the source variable.
    """
    try:
        source.value
        if source.value is None:
            raise AttributeError("Value is None")
        return write_urls_to_file(source.value)
    except AttributeError:
        try:
            source.selected
            return source.selected
        except AttributeError:
            return None


def choose_folder(start_path: str = ".", folder_type: str = ""):
    # Specify the output folder
    fc = FileChooser(start_path)
    fc.title = f"Choose location of {folder_type}"
    display(fc)
    return fc


def write_urls_to_file(movie_list: list, filepath: str = "/tmp/temp.txt"):
    """
    > This function takes a list of movie urls and writes them to a file
    so that they can be passed to the detect method of the ML models

    :param movie_list: list
    :type movie_list: list
    :param filepath: The path to the file to write the urls to, defaults to /tmp/temp.txt
    :type filepath: str (optional)
    :return: The filepath of the file that was written to.
    """
    try:
        iter(movie_list)
    except TypeError:
        logging.error(
            "No source movies found in selected path or path is empty. Please fix the previous selection"
        )
        return
    with open(filepath, "w") as fp:
        fp.write("\n".join(movie_list))
    return filepath


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


def choose_project(
    projects_csv: str = "../kso_utils/kso_utils/db_starter/projects_list.csv",
):
    """
    > This function takes a csv file with a list of projects and returns a dropdown menu with the
    projects listed

    :param projects_csv: str = "../kso_utils/db_starter/projects_list.csv", defaults to ../kso_utils/db_starter/projects_list.csv
    :type projects_csv: str (optional)
    :return: A dropdown widget with the project names as options.
    """

    # Check path to the list of projects is a csv
    if os.path.exists(projects_csv) and not projects_csv.endswith(".csv"):
        logging.error("A csv file was not selected. Please try again.")

    # If list of projects doesn't exist retrieve it from github
    if not os.path.exists(projects_csv):
        projects_csv = "https://github.com/ocean-data-factory-sweden/kso_utils/blob/main/kso_utils/db_starter/projects_list.csv?raw=true"

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


def select_retrieve_info():
    """
    Display a widget that allows to select whether to retrieve the last available information,
    or to request the latest information.

    :return: an interactive widget object with the value of the boolean

    """

    def generate_export(retrieve_option):
        if retrieve_option == "No, just download the last available information":
            generate = False

        elif retrieve_option == "Yes":
            generate = True

        return generate

    latest_info = interactive(
        generate_export,
        retrieve_option=widgets.RadioButtons(
            options=["Yes", "No, just download the last available information"],
            value="No, just download the last available information",
            layout={"width": "max-content"},
            description="Do you want to request the most up-to-date Zooniverse information?",
            disabled=False,
            style={"description_width": "initial"},
        ),
    )

    display(latest_info)
    display(
        HTML(
            """<font size="2px">If yes, a new data export will be requested and generated with the latest information of Zooniverse (this may take some time)<br>
    Otherwise, the latest available export will be downloaded (some recent information may be missing!!).<br><br>
    If the waiting time for the generation of a new data export ends, the last available information will be retrieved. However, that information <br>
    will probably correspond to the newly generated export.
    </font>"""
        )
    )

    return latest_info


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
        ensure_option=False,
        value=None,
        disabled=False,
        layout=widgets.Layout(width="50%"),
        style={"description_width": "initial"},
    )

    display(select_movie_widget)
    return select_movie_widget


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


def single_wait_for_change(widget, value):
    future = asyncio.Future()

    def getvalue(change):
        future.set_result(change.new)
        widget.unobserve(getvalue, value)

    widget.observe(getvalue, value)
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
