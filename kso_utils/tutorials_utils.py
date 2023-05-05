# base imports
import os
import re
import time
import json
import cv2
import pandas as pd
import numpy as np
import logging
import subprocess
import datetime
import random
import imagesize
import requests
import ffmpeg as ffmpeg_python
from tqdm import tqdm
from base64 import b64encode
from io import BytesIO
from urllib.parse import urlparse
from csv_diff import compare, load_csv
from pathlib import Path
from PIL import Image as PILImage, ImageDraw

# widget imports
import ipysheet
import ipywidgets as widgets
from ipyfilechooser import FileChooser
from IPython.display import HTML, display, clear_output
from ipywidgets import interactive, Layout
import asyncio

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

############ CSV/iPysheet FUNCTIONS ################


def log_meta_changes(
    self,
    meta_key: str,
    new_sheet_df: pd.DataFrame,
):
    """Records changes to csv files in log file (json format)"""

    diff = {
        "timestamp": int(time.time()),
        "change_info": compare(
            {
                int(k): v
                for k, v in pd.read_csv(self.db_info[meta_key]).to_dict("index").items()
            },
            {int(k): v for k, v in new_sheet_df.to_dict("index").items()},
        ),
    }

    if len(diff) == 0:
        logging.info("No changes were logged")
        return

    else:
        try:
            with open(Path(self.project.csv_folder, "change_log.json"), "r+") as f:
                try:
                    existing_data = json.load(f)
                except json.decoder.JSONDecodeError:
                    existing_data = []
                existing_data.append(diff)
                f.seek(0)
                json.dump(existing_data, f)
        except FileNotFoundError:
            with open(Path(self.project.csv_folder, "change_log.json"), "w") as f:
                json.dump([diff], f)
        logging.info(
            f"Changelog updated at: {Path(self.project.csv_folder, 'change_log.json')}"
        )
        return


def open_csv(
    df: pd.DataFrame, df_range_rows: widgets.Widget, df_range_columns: widgets.Widget
):
    """
    > This function loads the dataframe with the information of interest, filters the range of rows and columns selected and then loads the dataframe into
    an ipysheet

    :param df: a pandas dataframe of the information of interest:
    :param df_range_rows: the rows range widget selection:
    :param df_range_columns: the columns range widget selection:
    :return: A (subset) dataframe with the information of interest and the same data in an interactive sheet
    """
    # Extract the first and last row to display
    range_start = int(df_range_rows.label[0])
    range_end = int(df_range_rows.label[1])

    # Extract the first and last columns to display
    if not len(df_range_columns.label) == 0:
        column_start = str(df_range_columns.label[0])
        column_end = str(df_range_columns.label[-1])
    else:
        column_start = df.columns[0]
        column_end = df.columns[-1]

    # Display the range of sites selected
    logging.info(f"Displaying # {range_start} to # {range_end}")
    logging.info(f"Displaying {column_start} to {column_end}")

    # Filter the dataframe based on the selection: rows and columns
    df_filtered_row = df.filter(items=range(range_start, range_end), axis=0)
    df_filtered = df_filtered_row.filter(items=df.columns, axis=1)

    # Load the df as ipysheet
    sheet = ipysheet.from_dataframe(df_filtered)

    return df_filtered, sheet


def display_changes(
    db_info_dict: dict, isheet: ipysheet.Sheet, df_filtered: pd.DataFrame
):
    """
    It takes the dataframe from the ipysheet and compares it to the dataframe from the local csv file.
    If there are any differences, it highlights them and returns the dataframe with the changes
    highlighted

    :param db_info_dict: a dictionary containing the database information
    :type db_info_dict: dict
    :param isheet: The ipysheet object that contains the data
    :param sites_df_filtered: a pandas dataframe with information of a range of sites
    :return: A tuple with the highlighted changes and the sheet_df
    """
    # Convert ipysheet to pandas
    sheet_df = ipysheet.to_dataframe(isheet)

    # Check the differences between the modified and original spreadsheets
    sheet_diff_df = pd.concat([df_filtered, sheet_df]).drop_duplicates(keep=False)

    # If changes in dataframes display them and ask the user to confirm them
    if sheet_diff_df.empty:
        logging.info("No changes were made.")
        return sheet_df, sheet_df
    else:
        # Retieve the column name of the id of interest (Sites, movies,..)
        id_col = [col for col in df_filtered.columns if "_id" in col][0]

        # Concatenate DataFrames and distinguish each frame with the keys parameter
        df_all = pd.concat(
            [df_filtered.set_index(id_col), sheet_df.set_index(id_col)],
            axis="columns",
            keys=["Origin", "Update"],
        )

        # Rearrange columns to have them next to each other
        df_final = df_all.swaplevel(axis="columns")[df_filtered.columns[1:]]

        # Create a function to highlight the changes
        def highlight_diff(data, color="yellow"):
            attr = "background-color: {}".format(color)
            other = data.xs("Origin", axis="columns", level=-1)
            return pd.DataFrame(
                np.where(data.ne(other, level=0), attr, ""),
                index=data.index,
                columns=data.columns,
            )

        # Return the df with the changes highlighted
        highlight_changes = df_final.style.apply(highlight_diff, axis=None)

        return highlight_changes, sheet_df


def select_sheet_range(db_info_dict: dict, orig_csv: str):
    """
    > This function loads the csv file of interest into a pandas dataframe and enables users to pick a range of rows and columns to display

    :param db_info_dict: a dictionary with the following keys:
    :param orig_csv: the original csv file name
    :type orig_csv: str
    :return: A dataframe with the sites information
    """

    # Load the csv with the information of interest
    df = pd.read_csv(db_info_dict[orig_csv])

    df_range_rows = widgets.SelectionRangeSlider(
        options=range(0, len(df.index) + 1),
        index=(0, len(df.index)),
        description="Rows to display",
        orientation="horizontal",
        layout=Layout(width="90%", padding="35px"),
        style={"description_width": "initial"},
    )

    display(df_range_rows)

    df_range_columns = widgets.SelectMultiple(
        options=df.columns,
        description="Columns",
        disabled=False,
        layout=Layout(width="50%", padding="35px"),
    )

    display(df_range_columns)

    return df, df_range_rows, df_range_columns


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


def choose_project(projects_csv: str = "../kso_utils/db_starter/projects_list.csv"):
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


def choose_movie_review():
    """
    This function creates a widget that allows the user to choose between two methods to review the
    movies.csv file.
    :return: The widget is being returned.
    """
    choose_movie_review_widget = widgets.RadioButtons(
        options=[
            "Basic: Automatic check for empty fps/duration and sampling start/end cells in the movies.csv",
            "Advanced: Basic + Check format and metadata of each movie",
        ],
        description="What method you want to use to review the movies:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(choose_movie_review_widget)
    return choose_movie_review_widget


def select_deployment(missing_from_csv: pd.DataFrame):
    """
    > This function takes a dataframe of missing files and returns a widget that allows the user to
    select the deployment of interest

    :param missing_from_csv: a dataframe of the files that are missing from the csv file
    :return: A widget object
    """
    if missing_from_csv.shape[0] > 0:
        # Widget to select the deployment of interest
        deployment_widget = widgets.SelectMultiple(
            options=missing_from_csv.deployment_folder.unique(),
            description="New deployment:",
            disabled=False,
            layout=Layout(width="50%"),
            style={"description_width": "initial"},
        )
        display(deployment_widget)
        return deployment_widget


def select_eventdate():
    # Select the date
    """
    > This function creates a date picker widget that allows the user to select a date.

    The function is called `select_eventdate()` and it returns a date picker widget.
    """
    date_widget = widgets.DatePicker(
        description="Date of deployment:",
        value=datetime.date.today(),
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(date_widget)
    return date_widget


def choose_new_videos_to_upload():
    """
    Simple widget for uploading videos from a file browser.
    returns the list of movies to be added.
    Supports multi-select file uploads
    """

    movie_list = []

    fc = FileChooser()
    fc.title = "First choose your directory of interest"
    " and then the movies you would like to upload"
    logging.info("Choose the file that you want to upload: ")

    def change_dir(chooser):
        sel.options = os.listdir(chooser.selected)
        fc.children[1].children[2].layout.display = "none"
        sel.layout.visibility = "visible"

    fc.register_callback(change_dir)

    sel = widgets.SelectMultiple(options=os.listdir(fc.selected))

    display(fc)
    display(sel)

    sel.layout.visibility = "hidden"

    button_add = widgets.Button(description="Add selected file")
    output_add = widgets.Output()

    logging.info(
        "Showing paths to the selected movies:\nRerun cell to reset\n--------------"
    )

    display(button_add, output_add)

    def on_button_add_clicked(b):
        with output_add:
            if sel.value is not None:
                for movie in sel.value:
                    if Path(movie).suffix in [".mp4", ".mov"]:
                        movie_list.append([Path(fc.selected, movie), movie])
                        logging.info(Path(fc.selected, movie))
                    else:
                        logging.error("Invalid file extension")
                    fc.reset()

    button_add.on_click(on_button_add_clicked)
    return movie_list


####################################################
############### SURVEY FUNCTIONS ###################
####################################################


def progress_handler(progress_info):
    print("{:.2f}".format(progress_info["percentage"]))


def record_encoder():
    """
    > This function creates a widget that asks for the name of the person encoding the survey
    information
    :return: The name of the person encoding the survey information
    """
    # Widget to record the encoder of the survey information
    EncoderName_widget = widgets.Text(
        placeholder="First and last name",
        description="Name of the person encoding this survey information:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(EncoderName_widget)
    return EncoderName_widget


def select_SurveyStartDate():
    """
    > This function creates a widget that allows the user to select a date
    :return: A widget that allows the user to select a date.
    """
    # Widget to record the start date of the survey
    SurveyStartDate_widget = widgets.DatePicker(
        description="Offical date when survey started as a research event",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(SurveyStartDate_widget)
    return SurveyStartDate_widget


def write_SurveyName():
    """
    > This function creates a widget that allows the user to enter a name for the survey
    :return: A widget object
    """
    # Widget to record the name of the survey
    SurveyName_widget = widgets.Text(
        placeholder="Baited Underwater Video Taputeranga Apr 2015",
        description="A name for this survey:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(SurveyName_widget)
    return SurveyName_widget


def select_OfficeName(OfficeName_options: list):
    """
    > This function creates a dropdown widget that allows the user to select the name of the DOC Office
    responsible for the survey

    :param OfficeName_options: a list of the names of the DOC offices that are responsible for the survey
    :return: The widget is being returned.
    """
    # Widget to record the name of the linked DOC Office
    OfficeName_widget = widgets.Dropdown(
        options=OfficeName_options,
        description="Department of Conservation Office responsible for this survey:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(OfficeName_widget)
    return OfficeName_widget


def write_ContractorName():
    """
    > This function creates a widget that allows the user to enter the name of the contractor
    :return: The widget is being returned.
    """
    # Widget to record the name of the contractor
    ContractorName_widget = widgets.Text(
        placeholder="No contractor",
        description="Person/company contracted to carry out the survey:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(ContractorName_widget)
    return ContractorName_widget


def write_ContractNumber():
    """
    > This function creates a widget that allows the user to enter the contract number for the survey
    :return: The widget is being returned.
    """
    # Widget to record the number of the contractor
    ContractNumber_widget = widgets.Text(
        description="Contract number for this survey:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(ContractNumber_widget)
    return ContractNumber_widget


def write_LinkToContract():
    """
    > This function creates a widget that allows the user to enter a hyperlink to the contract related
    to the survey
    :return: The widget
    """
    # Widget to record the link to the contract
    LinkToContract_widget = widgets.Text(
        description="Hyperlink to the DOCCM for the contract related to this survey:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(LinkToContract_widget)
    return LinkToContract_widget


def write_SurveyLeaderName():
    """
    > This function creates a widget that allows the user to enter the name of the person in charge of
    the survey
    :return: The widget is being returned.
    """
    # Widget to record the name of the survey leader
    SurveyLeaderName_widget = widgets.Text(
        placeholder="First and last name",
        description="Name of the person in charge of this survey:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(SurveyLeaderName_widget)
    return SurveyLeaderName_widget


def select_LinkToMarineReserve(reserves_available: list):
    """
    > This function creates a dropdown widget that allows the user to select the name of the Marine
    Reserve that the survey is linked to

    :param reserves_available: a list of the names of the Marine Reserves that are available to be linked to the survey
    :return: The name of the Marine Reserve linked to the survey
    """
    # Widget to record the name of the linked Marine Reserve
    LinkToMarineReserve_widget = widgets.Dropdown(
        options=reserves_available,
        description="Marine Reserve linked to the survey:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(LinkToMarineReserve_widget)
    return LinkToMarineReserve_widget


def select_FishMultiSpecies():
    """
    > This function creates a widget that allows the user to select whether the survey is a single
    species survey or not
    :return: A widget that can be used to select a single species or multiple species
    """

    # Widget to record if survey is single species
    def FishMultiSpecies_to_true_false(FishMultiSpecies_value):
        if FishMultiSpecies_value == "Yes":
            return False
        else:
            return True

    w = interactive(
        FishMultiSpecies_to_true_false,
        FishMultiSpecies_value=widgets.Dropdown(
            options=["No", "Yes"],
            description="Does this survey look at a single species?",
            disabled=False,
            layout=Layout(width="95%"),
            style={"description_width": "initial"},
        ),
    )
    display(w)
    return w


def select_StratifiedBy(StratifiedBy_choices: list):
    """
    > This function creates a dropdown widget that allows the user to select the stratified factors for
    the sampling design

    :param StratifiedBy_choices: a list of choices for the dropdown menu
    :return: A widget object
    """
    # Widget to record if survey was stratified by any factor
    StratifiedBy_widget = widgets.Dropdown(
        options=StratifiedBy_choices,
        description="Stratified factors for the sampling design",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(StratifiedBy_widget)
    return StratifiedBy_widget


def select_IsLongTermMonitoring():
    """
    > This function creates a widget that allows the user to select whether the survey is part of a
    long-term monitoring
    :return: A widget object
    """

    # Widget to record if survey is part of long term monitoring
    def IsLongTermMonitoring_to_true_false(IsLongTermMonitoring_value):
        if IsLongTermMonitoring_value == "No":
            return False
        else:
            return True

    w = interactive(
        IsLongTermMonitoring_to_true_false,
        IsLongTermMonitoring_value=widgets.Dropdown(
            options=["Yes", "No"],
            description="Is the survey part of a long-term monitoring?",
            disabled=False,
            layout=Layout(width="95%"),
            style={"description_width": "initial"},
        ),
    )
    display(w)
    return w


def select_SiteSelectionDesign(site_selection_options: list):
    """
    This function creates a dropdown widget that allows the user to select the site selection design of
    the survey

    :param site_selection_options: a list of strings that are the options for the dropdown menu
    :return: A widget
    """
    # Widget to record the site selection of the survey
    SiteSelectionDesign_widget = widgets.Dropdown(
        options=site_selection_options,
        description="What was the design for site selection?",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(SiteSelectionDesign_widget)
    return SiteSelectionDesign_widget


def select_UnitSelectionDesign(unit_selection_options: list):
    """
    > This function creates a dropdown widget that allows the user to select the design for site
    selection

    :param unit_selection_options: list
    :type unit_selection_options: list
    :return: A widget that allows the user to select the unit selection design of the survey.
    """
    # Widget to record the unit selection of the survey
    UnitSelectionDesign_widget = widgets.Dropdown(
        options=unit_selection_options,
        description="What was the design for unit selection?",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(UnitSelectionDesign_widget)
    return UnitSelectionDesign_widget


def select_RightsHolder(RightsHolder_options: list):
    """
    > This function creates a dropdown widget that allows the user to select the type of right holder of
    the survey

    :param RightsHolder_options: a list of options for the dropdown menu
    :return: A widget
    """
    # Widget to record the type of right holder of the survey
    RightsHolder_widget = widgets.Dropdown(
        options=RightsHolder_options,
        description="Person(s) or organization(s) owning or managing rights over the resource",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(RightsHolder_widget)
    return RightsHolder_widget


def select_AccessRights():
    """
    > This function creates a widget that asks the user to enter information about who can access the
    resource
    :return: A widget object
    """
    # Widget to record information about who can access the resource
    AccessRights_widget = widgets.Text(
        placeholder="",
        description="Who can access the resource?",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(AccessRights_widget)
    return AccessRights_widget


def write_SurveyVerbatim():
    """
    > This function creates a widget that allows the user to enter a description of the survey design
    and objectives
    :return: A widget
    """
    # Widget to record description of the survey design and objectives
    SurveyVerbatim_widget = widgets.Textarea(
        placeholder="",
        description="Provide an exhaustive description of the survey design and objectives",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(SurveyVerbatim_widget)
    return SurveyVerbatim_widget


def select_BUVType(BUVType_choices: list):
    """
    > This function creates a dropdown widget that allows the user to select the type of BUV used for
    the survey

    :param BUVType_choices: list
    :type BUVType_choices: list
    :return: A widget
    """
    # Widget to record the type of BUV
    BUVType_widget = widgets.Dropdown(
        options=BUVType_choices,
        description="Type of BUV used for the survey:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(BUVType_widget)
    return BUVType_widget


def write_LinkToPicture():
    """
    > This function creates a text box for the user to enter the link to the DOCCM folder for the survey
    photos
    :return: The widget is being returned.
    """
    # Widget to record the link to the pictures
    LinkToPicture_widget = widgets.Text(
        description="Hyperlink to the DOCCM folder for this survey photos:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(LinkToPicture_widget)
    return LinkToPicture_widget


def write_Vessel():
    """
    This function creates a widget that allows the user to enter the name of the vessel that deployed
    the unit
    :return: The name of the vessel used to deploy the unit.
    """
    # Widget to record the name of the vessel
    Vessel_widget = widgets.Text(
        description="Vessel used to deploy the unit:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(Vessel_widget)
    return Vessel_widget


def write_LinkToFieldSheets():
    """
    **write_LinkToFieldSheets()**: This function creates a text box for the user to enter a hyperlink to
    the DOCCM for the field sheets used to gather the survey information
    :return: The text box widget.
    """
    LinkToFieldSheets = widgets.Text(
        description="Hyperlink to the DOCCM for the field sheets used to gather the survey information:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(LinkToFieldSheets)
    return LinkToFieldSheets


def write_LinkReport01():
    """
    > This function creates a text box for the user to enter a hyperlink to the first (of up to four)
    DOCCM report related to these data
    :return: The text box.
    """
    LinkReport01 = widgets.Text(
        description="Hyperlink to the first (of up to four) DOCCM report related to these data:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(LinkReport01)
    return LinkReport01


def write_LinkReport02():
    """
    **write_LinkReport02()**: This function creates a text box for the user to enter a hyperlink to the
    second DOCCM report related to these data
    :return: The text box widget.
    """
    LinkReport02 = widgets.Text(
        description="Hyperlink to the second DOCCM report related to these data:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(LinkReport02)
    return LinkReport02


def write_LinkReport03():
    """
    **write_LinkReport03()**: This function creates a text box for the user to enter a hyperlink to the
    third DOCCM report related to these data
    :return: The text box widget.
    """
    LinkReport03 = widgets.Text(
        description="Hyperlink to the third DOCCM report related to these data:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(LinkReport03)
    return LinkReport03


def write_LinkReport04():
    """
    **write_LinkReport04()**: This function creates a text box for the user to enter a hyperlink to the
    fourth DOCCM report related to these data
    :return: The text box widget.
    """
    LinkReport04 = widgets.Text(
        description="Hyperlink to the fourth DOCCM report related to these data:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(LinkReport04)
    return LinkReport04


def write_LinkToOriginalData():
    """
    > This function creates a text box that allows the user to enter a hyperlink to the DOCCM for the
    spreadsheet where these data were intially encoded
    :return: A text box with a description.
    """
    LinkToOriginalData = widgets.Text(
        description="Hyperlink to the DOCCM for the spreadsheet where these data were intially encoded:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(LinkToOriginalData)
    return LinkToOriginalData


def get_survey_name(survey_i):
    """
    If the survey is new, save the responses for the new survey as a dataframe. If the survey is
    existing, return the name of the survey

    :param survey_i: the survey object
    :return: The name of the survey
    """
    # If new survey, review details and save changes in survey csv server
    if isinstance(survey_i.result, dict):
        # Save the responses for the new survey as a dataframe
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
        survey_name = new_survey_row_dict["SurveyName"]

    # If existing survey print the info for the pre-existing survey
    else:
        # Return the name of the survey
        survey_name = survey_i.result.value

    return survey_name


def select_eventdate(survey_row: pd.DataFrame):
    """
    > This function creates a date picker widget that allows the user to select the date of the survey.
    The default date is the beginning of the survey

    :param survey_row: a dataframe containing survey information
    :return: A widget object
    """
    # Set the beginning of the survey as default date
    default_date = pd.Timestamp(survey_row["SurveyStartDate"].values[0]).to_pydatetime()

    # Select the date
    date_widget = widgets.DatePicker(
        description="Date of deployment:",
        value=default_date,
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(date_widget)
    return date_widget


def select_SamplingStart(duration_i: int):
    # Select the start of the survey
    surv_start = interactive(
        to_hhmmss,
        seconds=widgets.IntSlider(
            value=0,
            min=0,
            max=duration_i,
            step=1,
            description="Survey starts (seconds):",
            layout=Layout(width="50%"),
            style={"description_width": "initial"},
        ),
    )
    display(surv_start)
    return surv_start


def select_SamplingEnd(duration_i: int):
    #     # Set default to 30 mins or max duration
    #     start_plus_30 = surv_start_i+(30*60)

    #     if start_plus_30>duration_i:
    #         default_end = duration_i
    #     else:
    #         default_end = start_plus_30

    # Select the end of the survey
    surv_end = interactive(
        to_hhmmss,
        seconds=widgets.IntSlider(
            value=duration_i,
            min=0,
            max=duration_i,
            step=1,
            description="Survey ends (seconds):",
            layout=Layout(width="50%"),
            style={"description_width": "initial"},
        ),
    )
    display(surv_end)
    return surv_end


def select_IsBadDeployment():
    """
    > This function creates a dropdown widget that allows the user to select whether or not the
    deployment is bad
    :return: A widget object
    """

    def deployment_to_true_false(deploy_value):
        if deploy_value == "No, it is a great video":
            return False
        else:
            return True

    w = interactive(
        deployment_to_true_false,
        deploy_value=widgets.Dropdown(
            options=["Yes, unfortunately it is marine crap", "No, it is a great video"],
            value="No, it is a great video",
            description="Is it a bad deployment?",
            disabled=False,
            layout=Layout(width="50%"),
            style={"description_width": "initial"},
        ),
    )
    display(w)
    return w


def write_ReplicateWithinSite():
    """
    This function creates a widget that allows the user to select the depth of the deployment
    :return: The value of the widget.
    """
    # Select the depth of the deployment
    ReplicateWithinSite_widget = widgets.BoundedIntText(
        value=0,
        min=0,
        max=1000,
        step=1,
        description="Number of the replicate within site (Field number of planned BUV station):",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(ReplicateWithinSite_widget)
    return ReplicateWithinSite_widget


# Select the person who recorded the deployment
def select_RecordedBy(existing_recorders: list):
    """
    This function takes a list of existing recorders and returns a widget that allows the user to select
    an existing recorder or enter a new one

    :param existing_recorders: a list of existing recorders
    :return: A widget with a dropdown menu that allows the user to select between two options, 'Existing' and 'New author'.
    """

    def f(Existing_or_new):
        if Existing_or_new == "Existing":
            RecordedBy_widget = widgets.Dropdown(
                options=existing_recorders,
                description="Existing recorder:",
                disabled=False,
                layout=Layout(width="50%"),
                style={"description_width": "initial"},
            )

        if Existing_or_new == "New author":
            RecordedBy_widget = widgets.Text(
                placeholder="First and last name",
                description="Recorder:",
                disabled=False,
                layout=Layout(width="50%"),
                style={"description_width": "initial"},
            )

        display(RecordedBy_widget)
        return RecordedBy_widget

    w = interactive(
        f,
        Existing_or_new=widgets.Dropdown(
            options=["Existing", "New author"],
            description="Deployment recorded by existing or new person:",
            disabled=False,
            layout=Layout(width="50%"),
            style={"description_width": "initial"},
        ),
    )
    display(w)
    return w


def select_DepthStrata():
    # Select the depth of the deployment
    deployment_DepthStrata = widgets.Text(
        placeholder="5-25m",
        description="Depth stratum within which the BUV unit was deployed:",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(deployment_DepthStrata)
    return deployment_DepthStrata


def select_Depth():
    # Select the depth of the deployment
    deployment_depth = widgets.BoundedIntText(
        value=0,
        min=0,
        max=100,
        step=1,
        description="Depth reading in meters at the time of BUV unit deployment:",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(deployment_depth)
    return deployment_depth


def select_UnderwaterVisibility(visibility_options: list):
    """
    > This function creates a dropdown menu with the options of the visibility of the water during the
    video deployment

    :param visibility_options: a list of options for the dropdown menu
    :return: The dropdown menu with the options for the water visibility of the video deployment.
    """
    UnderwaterVisibility = widgets.Dropdown(
        options=visibility_options,
        description="Water visibility of the video deployment:",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(UnderwaterVisibility)
    return UnderwaterVisibility


def deployment_TimeIn():
    # Select the TimeIn
    TimeIn_widget = widgets.TimePicker(
        description="Time in of the deployment:",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(TimeIn_widget)
    return TimeIn_widget


def deployment_TimeOut():
    # Select the TimeOut
    TimeOut_widget = widgets.TimePicker(
        description="Time out of the deployment:",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(TimeOut_widget)
    return TimeOut_widget


# Write a comment about the deployment
def write_NotesDeployment():
    # Create the comment widget
    comment_widget = widgets.Text(
        placeholder="Type comment",
        description="Comment:",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(comment_widget)
    return comment_widget


def select_DeploymentDurationMinutes():
    # Select the theoretical duration of the deployment
    DeploymentDurationMinutes = widgets.BoundedIntText(
        value=0,
        min=0,
        max=60,
        step=1,
        description="Theoretical minimum soaking time for the unit (mins):",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(DeploymentDurationMinutes)
    return DeploymentDurationMinutes


def write_Habitat():
    # Widget to record the type of habitat
    Habitat_widget = widgets.Text(
        placeholder="Make and model",
        description="Describe the nature of the seabed (mud, sand, gravel, cobbles, rocky reef, kelp forest…)",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(Habitat_widget)
    return Habitat_widget


def write_NZMHCS_Abiotic():
    # Widget to record the type of NZMHCS_Abiotic
    NZMHCS_Abiotic_widget = widgets.Text(
        placeholder="0001",
        description="Write the Abiotic New Zealand Marine Habitat Classification number (Table 5)",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(NZMHCS_Abiotic_widget)
    return NZMHCS_Abiotic_widget


def write_NZMHCS_Biotic():
    # Widget to record the type of NZMHCS_Biotic
    NZMHCS_Biotic_widget = widgets.Text(
        placeholder="0001",
        description="Write the Biotic New Zealand Marine Habitat Classification number (Table 6)",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(NZMHCS_Biotic_widget)
    return NZMHCS_Biotic_widget


# Widget to record the level of the tide
def select_TideLevel(TideLevel_choices: list):
    TideLevel_widget = widgets.Dropdown(
        options=TideLevel_choices,
        description="Tidal level at the time of sampling:",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(TideLevel_widget)
    return TideLevel_widget


# Widget to record the weather
def write_Weather():
    Weather_widget = widgets.Text(
        description="Describe the weather for the survey:",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(Weather_widget)
    return Weather_widget


def select_CameraModel(CameraModel_choices: list):
    # Widget to record the type of camera
    CameraModel_widget = widgets.Dropdown(
        options=CameraModel_choices,
        description="Select the make and model of camera used",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(CameraModel_widget)
    return CameraModel_widget


def write_LensModel():
    # Widget to record the camera settings
    LensModel_widget = widgets.Text(
        placeholder="Wide lens, 1080x1440",
        description="Describe the camera lens and settings",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(LensModel_widget)
    return LensModel_widget


def write_BaitSpecies():
    # Widget to record the type of bait used
    BaitSpecies_widget = widgets.Text(
        placeholder="Pilchard",
        description="Species that was used as bait for the deployment",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(BaitSpecies_widget)
    return BaitSpecies_widget


def select_BaitAmount():
    # Widget to record the amount of bait used
    BaitAmount_widget = widgets.BoundedIntText(
        value=500,
        min=100,
        max=1000,
        step=1,
        description="Amount of bait used (g):",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(BaitAmount_widget)
    return BaitAmount_widget


# Display in hours, minutes and seconds
def to_hhmmss(seconds):
    print("Time selected:", datetime.timedelta(seconds=seconds))
    return seconds


def select_clip_length():
    """
    > This function creates a dropdown widget that allows the user to select the length of the clips
    :return: The widget is being returned.
    """
    # Widget to record the length of the clips
    ClipLength_widget = widgets.Dropdown(
        options=[10, 5],
        value=10,
        description="Length of clips:",
        style={"description_width": "initial"},
        ensure_option=True,
        disabled=False,
    )

    return ClipLength_widget


# Function to extract the videos
def extract_example_clips(
    output_clip_path: str, start_time_i: int, clip_length: int, movie_path: str
):
    """
    > Extracts a clip from a movie file, and saves it to a new file

    :param output_clip_path: The path to the output clip
    :param start_time_i: The start time of the clip in seconds
    :param clip_length: The length of the clip in seconds
    :param movie_path: the path to the movie file
    """

    # Extract the clip
    if not os.path.exists(output_clip_path):
        subprocess.call(
            [
                "ffmpeg",
                "-ss",
                str(start_time_i),
                "-t",
                str(clip_length),
                "-i",
                str(movie_path),
                "-c",
                "copy",
                "-an",  # removes the audio
                "-force_key_frames",
                "1",
                str(output_clip_path),
            ]
        )

        os.chmod(output_clip_path, 0o777)


def check_clip_size(clips_list: list):
    """
    > This function takes a list of file paths and returns a dataframe with the file path and size of
    each file. If the size is too large, we suggest compressing them as a first step.

    :param clips_list: list of file paths to the clips you want to check
    :type clips_list: list
    :return: A dataframe with the file path and size of each clip
    """

    # Get list of files with size
    if clips_list is None:
        logging.error("No clips found.")
        return None
    files_with_size = [
        (file_path, os.path.getsize(file_path) / float(1 << 20))
        for file_path in clips_list
    ]

    df = pd.DataFrame(files_with_size, columns=["File_path", "Size"])

    if df["Size"].ge(8).any():
        logging.info(
            "Clips are too large (over 8 MB) to be uploaded to Zooniverse. Compress them!"
        )
        return df
    else:
        logging.info(
            "Clips are a good size (below 8 MB). Ready to be uploaded to Zooniverse"
        )
        return df


class clip_modification_widget(widgets.VBox):
    def __init__(self):
        """
        The function creates a widget that allows the user to select which modifications to run
        """
        self.widget_count = widgets.IntText(
            description="Number of modifications:",
            display="flex",
            flex_flow="column",
            align_items="stretch",
            style={"description_width": "initial"},
        )
        self.bool_widget_holder = widgets.HBox(
            layout=widgets.Layout(
                width="100%", display="inline-flex", flex_flow="row wrap"
            )
        )
        children = [
            self.widget_count,
            self.bool_widget_holder,
        ]
        self.widget_count.observe(self._add_bool_widgets, names=["value"])
        super().__init__(children=children)

    def _add_bool_widgets(self, widg):
        num_bools = widg["new"]
        new_widgets = []
        for _ in range(num_bools):
            new_widget = select_modification()
            for wdgt in [new_widget]:
                wdgt.description = wdgt.description + f" #{_}"
            new_widgets.extend([new_widget])
        self.bool_widget_holder.children = tuple(new_widgets)

    @property
    def checks(self):
        return {w.description: w.value for w in self.bool_widget_holder.children}


def select_modification():
    """
    This function creates a dropdown widget that allows the user to select a clip modification
    :return: A widget that allows the user to select a clip modification.
    """
    # Widget to select the clip modification

    clip_modifications = {
        "Color_correction": {
            "filter": ".filter('curves', '0/0 0.396/0.67 1/1', \
                                        '0/0 0.525/0.451 1/1', \
                                        '0/0 0.459/0.517 1/1')"
        }
        # borrowed from https://www.element84.com/blog/color-correction-in-space-and-at-sea
        ,
        "Zoo_low_compression": {
            "crf": "25",
            "bv": "7",
        },
        "Zoo_medium_compression": {
            "crf": "27",
            "bv": "6",
        },
        "Zoo_high_compression": {
            "crf": "30",
            "bv": "5",
        },
        "Blur_sensitive_info": {
            "filter": ".drawbox(0, 0, 'iw', 'ih*(15/100)', color='black' \
                            ,thickness='fill').drawbox(0, 'ih*(95/100)', \
                            'iw', 'ih*(15/100)', color='black', thickness='fill')",
            "None": {},
        },
    }

    select_modification_widget = widgets.Dropdown(
        options=[(a, b) for a, b in clip_modifications.items()],
        description="Select modification:",
        ensure_option=True,
        disabled=False,
        style={"description_width": "initial"},
    )

    # display(select_modification_widget)
    return select_modification_widget


def modify_clips(
    clip_i: str, modification_details: dict, output_clip_path: str, gpu_available: bool
):
    """
    > This function takes in a clip, a dictionary of modification details, and an output path, and then
    modifies the clip using the details provided

    :param clip_i: the path to the clip to be modified
    :param modification_details: a dictionary of the modifications to be made to the clip
    :param output_clip_path: The path to the output clip
    :param gpu_available: If you have a GPU, set this to True. If you don't, set it to False
    """
    if gpu_available:
        # Unnest the modification detail dict
        df = pd.json_normalize(modification_details, sep="_")
        b_v = df.filter(regex="bv$", axis=1).values[0][0] + "M"

        subprocess.call(
            [
                "ffmpeg",
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-i",
                clip_i,
                "-c:a",
                "copy",
                "-c:v",
                "h264_nvenc",
                "-b:v",
                b_v,
                output_clip_path,
            ]
        )

    else:
        # Set up input prompt
        init_prompt = f"ffmpeg_python.input('{clip_i}')"
        default_output_prompt = f".output('{output_clip_path}', crf=20, pix_fmt='yuv420p', vcodec='libx264')"
        full_prompt = init_prompt
        mod_prompt = ""

        # Set up modification
        for transform in modification_details.values():
            if "filter" in transform:
                mod_prompt += transform["filter"]
            else:
                # Unnest the modification detail dict
                df = pd.json_normalize(modification_details, sep="_")
                crf = df.filter(regex="crf$", axis=1).values[0][0]
                out_prompt = f".output('{output_clip_path}', crf={crf}, preset='veryfast', pix_fmt='yuv420p', vcodec='libx264')"

        if len(mod_prompt) > 0:
            full_prompt += mod_prompt
        if out_prompt:
            full_prompt += out_prompt
        else:
            full_prompt += default_output_prompt

        # Run the modification
        try:
            eval(full_prompt).run(capture_stdout=True, capture_stderr=True)
            os.chmod(output_clip_path, 0o777)
        except ffmpeg_python.Error as e:
            logging.info("stdout:", e.stdout.decode("utf8"))
            logging.info("stderr:", e.stderr.decode("utf8"))
            raise e

    logging.info(f"Clip {clip_i} modified successfully")


# Display the clips side-by-side
def view_clips(example_clips: list, modified_clip_path: str):
    """
    > This function takes in a list of example clips and a path to a modified clip, and returns a widget
    that displays the original and modified clips side-by-side

    :param example_clips: a list of paths to the original clips
    :param modified_clip_path: The path to the modified clip you want to view
    :return: A widget that displays the original and modified videos side-by-side.
    """

    # Get the path of the modified clip selected
    example_clip_name = os.path.basename(modified_clip_path).replace("modified_", "")
    example_clip_path = next(
        filter(lambda x: os.path.basename(x) == example_clip_name, example_clips), None
    )

    # Get the extension of the video
    extension = Path(example_clip_path).suffix

    # Open original video
    vid1 = open(example_clip_path, "rb").read()
    wi1 = widgets.Video(value=vid1, format=extension, width=400, height=500)

    # Open modified video
    vid2 = open(modified_clip_path, "rb").read()
    wi2 = widgets.Video(value=vid2, format=extension, width=400, height=500)

    # Display videos side-by-side
    a = [wi1, wi2]
    wid = widgets.HBox(a)

    return wid


def compare_clips(example_clips: list, modified_clips: list):
    """
    > This function allows you to select a clip from the modified clips and displays the original and
    modified clips side by side

    :param example_clips: The original clips
    :param modified_clips: The list of clips that you want to compare to the original clips
    """

    # Add "no movie" option to prevent conflicts
    modified_clips = np.append(modified_clips, "0 No movie")

    clip_path_widget = widgets.Dropdown(
        options=tuple(modified_clips),
        description="Select original clip:",
        ensure_option=True,
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )

    main_out = widgets.Output()
    display(clip_path_widget, main_out)

    # Display the original and modified clips
    def on_change(change):
        with main_out:
            clear_output()
            if change["new"] == "0 No movie":
                logging.info("It is OK to modify the clips again")
            else:
                a = view_clips(example_clips, change["new"])
                display(a)

    clip_path_widget.observe(on_change, names="value")


def review_clip_selection(clip_selection, movie_i: str, clip_modification):
    """
    > This function reviews the clips that will be created from the movie selected

    :param clip_selection: the object that contains the results of the clip selection
    :param movie_i: the movie you want to create clips from
    :param clip_modification: The modification that will be applied to the clips
    """
    start_trim = clip_selection.kwargs["clips_range"][0]
    end_trim = clip_selection.kwargs["clips_range"][1]

    # Review the clips that will be created
    logging.info(
        f"You are about to create {round(clip_selection.result)} clips from {movie_i}"
    )
    logging.info(
        f"starting at {datetime.timedelta(seconds=start_trim)} and ending at {datetime.timedelta(seconds=end_trim)}"
    )
    logging.info(f"The modification selected is {clip_modification}")


# Func to expand seconds
def expand_list(df: pd.DataFrame, list_column: str, new_column: str):
    """
    We take a dataframe with a column that contains lists, and we expand that column into a new
    dataframe with a new column that contains the items in the list

    :param df: the dataframe you want to expand
    :param list_column: the column that contains the list
    :param new_column: the name of the new column that will be created
    :return: A dataframe with the list column expanded into a new column.
    """
    lens_of_lists = df[list_column].apply(len)
    origin_rows = range(df.shape[0])
    destination_rows = np.repeat(origin_rows, lens_of_lists)
    non_list_cols = [idx for idx, col in enumerate(df.columns) if col != list_column]
    expanded_df = df.iloc[destination_rows, non_list_cols].copy()
    expanded_df[new_column] = [item for items in df[list_column] for item in items]
    expanded_df.reset_index(inplace=True, drop=True)
    return expanded_df


# Function to extract the videos
def extract_clips(
    movie_path: str,
    clip_length: int,
    upl_second_i: int,
    output_clip_path: str,
    modification_details: dict,
    gpu_available: bool,
):
    """
    This function takes in a movie path, a clip length, a starting second index, an output clip path, a
    dictionary of modification details, and a boolean indicating whether a GPU is available. It then
    extracts a clip from the movie, and applies the modifications specified in the dictionary.

    The function is written in such a way that it can be used to extract clips from a movie, and apply
    modifications to the clips.

    :param movie_path: The path to the movie file
    :param clip_length: The length of the clip in seconds
    :param upl_second_i: The second in the video to start the clip
    :param output_clip_path: The path to the output clip
    :param modification_details: a dictionary of dictionaries, where each dictionary contains the
           details of a modification to be made to the video. The keys of the dictionary are the names of the
           modifications, and the values are dictionaries containing the details of the modification.
    :param gpu_available: If you have a GPU, set this to True. If you don't, set it to False
    """
    if not modification_details and gpu_available:
        # Create clips without any modification
        subprocess.call(
            [
                "ffmpeg",
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-ss",
                str(upl_second_i),
                "-t",
                str(clip_length),
                "-i",
                movie_path,
                "-threads",
                "4",
                "-an",  # removes the audio
                "-c:a",
                "copy",
                "-c:v",
                "h264_nvenc",
                str(output_clip_path),
            ]
        )
        os.chmod(str(output_clip_path), 0o777)

    elif modification_details and gpu_available:
        # Unnest the modification detail dict
        df = pd.json_normalize(modification_details, sep="_")
        b_v = df.filter(regex="bv$", axis=1).values[0][0] + "M"

        subprocess.call(
            [
                "ffmpeg",
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-ss",
                str(upl_second_i),
                "-t",
                str(clip_length),
                "-i",
                movie_path,
                "-threads",
                "4",
                "-an",  # removes the audio
                "-c:a",
                "copy",
                "-c:v",
                "h264_nvenc",
                "-b:v",
                b_v,
                str(output_clip_path),
            ]
        )
        os.chmod(str(output_clip_path), 0o777)
    else:
        # Set up input prompt
        init_prompt = f"ffmpeg_python.input('{movie_path}')"
        full_prompt = init_prompt
        mod_prompt = ""
        output_prompt = ""
        def_output_prompt = f".output('{str(output_clip_path)}', ss={str(upl_second_i)}, t={str(clip_length)}, movflags='+faststart', crf=20, pix_fmt='yuv420p', vcodec='libx264')"

        # Set up modification
        for transform in modification_details.values():
            if "filter" in transform:
                mod_prompt += transform["filter"]

            else:
                # Unnest the modification detail dict
                df = pd.json_normalize(modification_details, sep="_")
                crf = df.filter(regex="crf$", axis=1).values[0][0]
                output_prompt = f".output('{str(output_clip_path)}', crf={crf}, ss={str(upl_second_i)}, t={str(clip_length)}, movflags='+faststart', preset='veryfast', pix_fmt='yuv420p', vcodec='libx264')"

        # Run the modification
        try:
            if len(mod_prompt) > 0:
                full_prompt += mod_prompt
            if len(output_prompt) > 0:
                full_prompt += output_prompt
            else:
                full_prompt += def_output_prompt
            eval(full_prompt).run(capture_stdout=True, capture_stderr=True)
            os.chmod(str(output_clip_path), 0o777)
        except ffmpeg_python.Error as e:
            logging.info("stdout:", e.stdout.decode("utf8"))
            logging.info("stderr:", e.stderr.decode("utf8"))
            raise e

        logging.info("Clips extracted successfully")


def remove_temp_clips(upload_to_zoo: pd.DataFrame):
    """
    > This function takes a dataframe of clips that are ready to be uploaded to the Zooniverse, and
    removes the temporary clips that were created in the previous step

    :param upload_to_zoo: a dataframe with the following columns:
    :type upload_to_zoo: pd.DataFrame
    """

    for temp_clip in upload_to_zoo["clip_path"].unique().tolist():
        os.remove(temp_clip)

    logging.info("Files removed successfully")


# Function to clean label (no non-alpha characters)
def clean_label(label_string: str):
    label_string = label_string.upper()
    label_string = label_string.replace(" ", "")
    pattern = r"[^A-Za-z0-9]+"
    cleaned_string = re.sub(pattern, "", label_string)
    return cleaned_string


def write_movie_frames(key_movie_df: pd.DataFrame, url: str):
    """
    Function to get a frame from a movie
    """
    # Read the movie on cv2 and prepare to extract frames
    cap = cv2.VideoCapture(url)

    if cap.isOpened():
        # Get the frame numbers for each movie the fps and duration
        for index, row in tqdm(key_movie_df.iterrows(), total=key_movie_df.shape[0]):
            # Create the folder to store the frames if not exist
            if not os.path.exists(row["frame_path"]):
                cap.set(1, row["frame_number"])
                ret, frame = cap.read()
                if frame is not None:
                    cv2.imwrite(row["frame_path"], frame)
                    os.chmod(row["frame_path"], 0o777)
                else:
                    cv2.imwrite(row["frame_path"], np.zeros((100, 100, 3), np.uint8))
                    os.chmod(row["frame_path"], 0o777)
                    logging.info(
                        f"No frame was extracted for {url} at frame {row['frame_number']}"
                    )
    else:
        logging.info("Missing movie", url)


class WidgetMaker(widgets.VBox):
    def __init__(self, workflows_df: pd.DataFrame):
        """
        The function creates a widget that allows the user to select which workflows to run

        :param workflows_df: the dataframe of workflows
        """
        self.workflows_df = workflows_df
        self.widget_count = widgets.BoundedIntText(
            value=0,
            min=0,
            max=100,
            description="Number of workflows:",
            display="flex",
            flex_flow="column",
            align_items="stretch",
            style={"description_width": "initial"},
        )
        self.bool_widget_holder = widgets.HBox(
            layout=widgets.Layout(
                width="70%", display="inline-flex", flex_flow="row wrap"
            )
        )
        children = [
            self.widget_count,
            self.bool_widget_holder,
        ]
        self.widget_count.observe(self._add_bool_widgets, names=["value"])
        super().__init__(children=children)

    def _add_bool_widgets(self, widg):
        num_bools = widg["new"]
        new_widgets = []
        for _ in range(num_bools):
            new_widget = choose_workflows(self.workflows_df)
            for wdgt in new_widget:
                wdgt.description = wdgt.description + f" #{_}"
            new_widgets.extend(new_widget)
        self.bool_widget_holder.children = tuple(new_widgets)

    @property
    def checks(self):
        return {w.description: w.value for w in self.bool_widget_holder.children}


def choose_w_version(workflows_df: pd.DataFrame, workflow_id: str):
    """
    It takes a workflow ID and returns a dropdown widget with the available versions of the workflow

    :param workflows_df: a dataframe containing the workflows available in the Galaxy instance
    :param workflow_id: The name of the workflow you want to run
    :return: A tuple containing the widget and the list of versions available.
    """

    # Estimate the versions of the workflow available
    versions_available = (
        workflows_df[workflows_df.display_name == workflow_id].version.unique().tolist()
    )

    if len(versions_available) > 1:
        # Display the versions of the workflow available
        w_version = widgets.Dropdown(
            options=list(map(float, versions_available)),
            value=float(versions_available[0]),
            description="Minimum workflow version:",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
            style={"description_width": "initial"},
        )

    else:
        raise ValueError("There are no versions available for this workflow.")

    # display(w_version)
    return w_version, list(map(float, versions_available))


def get_workflow_ids(workflows_df: pd.DataFrame, workflow_names: list):
    # The function that takes a list of workflow names and returns a list of workflow
    # ids.
    return [
        workflows_df[workflows_df.display_name == wf_name].workflow_id.unique()[0]
        for wf_name in workflow_names
    ]


def choose_agg_parameters(subject_type: str = "clip", full_description: bool = True):
    """
    > This function creates a set of sliders that allow you to set the parameters for the aggregation
    algorithm

    :param subject_type: The type of subject you are aggregating. This can be either "frame" or "video"
    :type subject_type: str
    :return: the values of the sliders.
        Aggregation threshold: (0-1) Minimum proportion of citizen scientists that agree in their classification of the clip/frame.
        Min numbers of users: Minimum number of citizen scientists that need to classify the clip/frame.
        Object threshold (0-1): Minimum proportion of citizen scientists that agree that there is at least one object in the frame.
        IOU Epsilon (0-1): Minimum area of overlap among the classifications provided by the citizen scientists so that they will be considered to be in the same cluster.
        Inter user agreement (0-1): The minimum proportion of users inside a given cluster that must agree on the frame annotation for it to be accepted.
    """
    agg_users = widgets.FloatSlider(
        value=0.8,
        min=0,
        max=1.0,
        step=0.1,
        description="Aggregation threshold:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
    )
    # Create HTML widget for description
    description_widget = HTML(
        f"<p>Minimum proportion of citizen scientists that agree in their classification of the {subject_type}.</p>"
    )
    # Display both widgets in a VBox
    display(agg_users)
    if full_description:
        display(description_widget)
    min_users = widgets.IntSlider(
        value=3,
        min=1,
        max=15,
        step=1,
        description="Min numbers of users:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
    )
    # Create HTML widget for description
    description_widget = HTML(
        f"<p>Minimum number of citizen scientists that need to classify the {subject_type}.</p>"
    )
    # Display both widgets in a VBox
    display(min_users)
    if full_description:
        display(description_widget)
    if subject_type == "frame":
        agg_obj = widgets.FloatSlider(
            value=0.8,
            min=0,
            max=1.0,
            step=0.1,
            description="Object threshold:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".1f",
            display="flex",
            flex_flow="column",
            align_items="stretch",
            style={"description_width": "initial"},
        )
        # Create HTML widget for description
        description_widget = HTML(
            "<p>Minimum proportion of citizen scientists that agree that there is at least one object in the frame.</p>"
        )
        # Display both widgets in a VBox
        display(agg_obj)
        if full_description:
            display(description_widget)
        agg_iou = widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1.0,
            step=0.1,
            description="IOU Epsilon:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".1f",
            display="flex",
            flex_flow="column",
            align_items="stretch",
            style={"description_width": "initial"},
        )
        # Create HTML widget for description
        description_widget = HTML(
            "<p>Minimum area of overlap among the citizen science classifications to be considered as being in the same cluster.</p>"
        )
        # Display both widgets in a VBox
        display(agg_iou)
        if full_description:
            display(description_widget)
        agg_iua = widgets.FloatSlider(
            value=0.8,
            min=0,
            max=1.0,
            step=0.1,
            description="Inter user agreement:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".1f",
            display="flex",
            flex_flow="column",
            align_items="stretch",
            style={"description_width": "initial"},
        )
        # Create HTML widget for description
        description_widget = HTML(
            "<p>The minimum proportion of users inside a given cluster that must agree on the frame annotation for it to be accepted.</p>"
        )
        # Display both widgets in a VBox
        display(agg_iua)
        if full_description:
            display(description_widget)
        return agg_users, min_users, agg_obj, agg_iou, agg_iua
    else:
        return agg_users, min_users


def choose_workflows(workflows_df: pd.DataFrame):
    """
    It creates a dropdown menu for the user to choose a workflow name, a dropdown menu for the user to
    choose a subject type, and a dropdown menu for the user to choose a workflow version

    :param workflows_df: a dataframe containing the workflows you want to choose from
    :type workflows_df: pd.DataFrame
    """

    layout = widgets.Layout(width="auto", height="40px")  # set width and height

    # Display the names of the workflows
    workflow_name = widgets.Dropdown(
        options=workflows_df.display_name.unique().tolist(),
        value=workflows_df.display_name.unique().tolist()[0],
        description="Workflow name:",
        disabled=False,
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
        layout=layout,
    )

    # Display the type of subjects
    subj_type = widgets.Dropdown(
        options=["frame", "clip"],
        value="clip",
        description="Subject type:",
        disabled=False,
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
        layout=layout,
    )

    workflow_version, versions = choose_w_version(workflows_df, workflow_name.value)

    def on_change(change):
        with out:
            if change["name"] == "value":
                clear_output()
                workflow_version.options = choose_w_version(
                    workflows_df, change["new"]
                )[1]
                workflow_name.observe(on_change)

    out = widgets.Output()
    display(out)

    workflow_name.observe(on_change)
    return workflow_name, subj_type, workflow_version


def extract_custom_frames(input_path, output_dir, num_frames=None, frame_skip=None):
    """
    This function extracts frames from a video file and saves them as JPEG images.

    :param input_path: The file path of the input movie file that needs to be processed
    :param output_dir: The directory where the extracted frames will be saved as JPEG files
    :param num_frames: The number of frames to extract from the input video. If this parameter is
    provided, the function will randomly select num_frames frames to extract from the video
    :param frame_skip: frame_skip is an optional parameter that determines how many frames to skip
    between extracted frames. For example, if frame_skip is set to 10, then every 10th frame will be
    extracted. If frame_skip is not provided, then all frames will be extracted
    """
    # Open the input movie file
    cap = cv2.VideoCapture(input_path)

    # Get base filename
    input_stem = Path(input_path).stem

    # Get the total number of frames in the movie
    num_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine which frames to extract based on the input parameters
    if num_frames is not None:
        frames_to_extract = random.sample(range(num_frames_total), num_frames)
    elif frame_skip is not None:
        frames_to_extract = range(0, num_frames_total, frame_skip)
    else:
        frames_to_extract = range(num_frames_total)

    output_files = []

    # Loop through the frames and extract the selected ones
    for frame_idx in frames_to_extract:
        # Set the frame index for the next frame to read
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read the next frame
        ret, frame = cap.read()

        if ret:
            # Construct the output filename for this frame
            output_filename = os.path.join(
                output_dir, f"{input_stem}_frame_{frame_idx}.jpg"
            )

            # Write the frame to a JPEG file
            cv2.imwrite(output_filename, frame)

            # Add output filename to list of files
            output_files.append(output_filename)

    # Release the video capture object
    cap.release()

    return pd.DataFrame(output_files, columns=["frame_path"])


# Function to specify the frame modification
def select_modification():
    # Widget to select the frame modification

    frame_modifications = {
        "Color_correction": {
            "filter": ".filter('curves', '0/0 0.396/0.67 1/1', \
                                        '0/0 0.525/0.451 1/1', \
                                        '0/0 0.459/0.517 1/1')"
        }
        # borrowed from https://www.element84.com/blog/color-correction-in-space-and-at-sea
        ,
        "Zoo_low_compression": {
            "crf": "25",
        },
        "Zoo_medium_compression": {
            "crf": "27",
        },
        "Zoo_high_compression": {
            "crf": "30",
        },
        "Blur_sensitive_info": {
            "filter": ".drawbox(0, 0, 'iw', 'ih*(15/100)', color='black' \
                            ,thickness='fill').drawbox(0, 'ih*(95/100)', \
                            'iw', 'ih*(15/100)', color='black', thickness='fill')",
            "None": {},
        },
    }

    select_modification_widget = widgets.Dropdown(
        options=[(a, b) for a, b in frame_modifications.items()],
        description="Select modification:",
        ensure_option=True,
        disabled=False,
        style={"description_width": "initial"},
    )

    display(select_modification_widget)
    return select_modification_widget


def check_frame_size(frame_paths: list):
    """
    It takes a list of file paths, gets the size of each file, and returns a dataframe with the file
    path and size of each file

    :param frame_paths: a list of paths to the frames you want to check
    :return: A dataframe with the file path and size of each frame.
    """

    # Get list of files with size
    files_with_size = [
        (file_path, os.stat(file_path).st_size) for file_path in frame_paths
    ]

    df = pd.DataFrame(files_with_size, columns=["File_path", "Size"])

    # Change bytes to MB
    df["Size"] = df["Size"] / 1000000

    if df["Size"].ge(1).any():
        logging.info(
            "Frames are too large (over 1 MB) to be uploaded to Zooniverse. Compress them!"
        )
        return df
    else:
        logging.info(
            "Frames are a good size (below 1 MB). Ready to be uploaded to Zooniverse"
        )
        return df


# Function to compare original to modified frames
def compare_frames(df):
    if not isinstance(df, pd.DataFrame):
        df = df.df

    # Save the paths of the clips
    original_frame_paths = df["frame_path"].unique()

    # Add "no movie" option to prevent conflicts
    original_frame_paths = np.append(original_frame_paths, "No frame")

    clip_path_widget = widgets.Dropdown(
        options=tuple(np.sort(original_frame_paths)),
        description="Select original frame:",
        ensure_option=True,
        disabled=False,
        layout=widgets.Layout(width="50%"),
        style={"description_width": "initial"},
    )

    main_out = widgets.Output()
    display(clip_path_widget, main_out)

    # Display the original and modified clips
    def on_change(change):
        with main_out:
            clear_output()
            if change["new"] == "No frame":
                print("It is OK to modify the frames again")
            else:
                a = view_frames(df, change["new"])
                display(a)

    clip_path_widget.observe(on_change, names="value")


# Display the frames using html
def view_frames(df: pd.DataFrame, frame_path: str):
    # Get path of the modified clip selected
    modified_frame_path = df[df["frame_path"] == frame_path].modif_frame_path.values[0]
    extension = os.path.splitext(frame_path)[1]

    img1 = open(frame_path, "rb").read()
    wi1 = widgets.Image(value=img1, format=extension, width=400, height=500)
    img2 = open(modified_frame_path, "rb").read()
    wi2 = widgets.Image(value=img2, format=extension, width=400, height=500)
    a = [wi1, wi2]
    wid = widgets.HBox(a)

    return wid


def aggregate_labels(raw_class_df: pd.DataFrame, agg_users: float, min_users: int):
    """
    > This function takes a dataframe of classifications and returns a dataframe of classifications that
    have been filtered by the number of users that classified each subject and the proportion of users
    that agreed on their annotations

    :param raw_class_df: the dataframe of all the classifications
    :param agg_users: the proportion of users that must agree on a classification for it to be included
           in the final dataset
    :param min_users: The minimum number of users that must have classified a subject for it to be
           included in the final dataset
    :return: a dataframe with the aggregated labels.
    """
    # Calculate the number of users that classified each subject
    raw_class_df["n_users"] = raw_class_df.groupby("subject_ids")[
        "classification_id"
    ].transform("nunique")

    # Select classifications with at least n different user classifications
    raw_class_df = raw_class_df[raw_class_df.n_users >= min_users].reset_index(
        drop=True
    )

    # Calculate the proportion of unique classifications (it can have multiple annotations) per subject
    raw_class_df["class_n"] = raw_class_df.groupby(["subject_ids", "label"])[
        "classification_id"
    ].transform("nunique")

    # Calculate the proportion of users that agreed on their annotations
    raw_class_df["class_prop"] = raw_class_df.class_n / raw_class_df.n_users

    # Select annotations based on agreement threshold
    agg_class_df = raw_class_df[raw_class_df.class_prop >= agg_users].reset_index(
        drop=True
    )

    # Calculate the proportion of unique classifications aggregated per subject
    agg_class_df["class_n_agg"] = agg_class_df.groupby(["subject_ids"])[
        "label"
    ].transform("nunique")

    return agg_class_df


def launch_table(agg_class_df: pd.DataFrame, subject_type: str):
    """
    It takes in a dataframe of aggregated classifications and a subject type, and returns a dataframe
    with the columns "subject_ids", "label", "how_many", and "first_seen"

    :param agg_class_df: the dataframe that you want to launch
    :param subject_type: "clip" or "subject"
    """
    if subject_type == "clip":
        a = agg_class_df[["subject_ids", "label", "how_many", "first_seen"]]
    else:
        a = agg_class_df

    return a


def process_frames(df: pd.DataFrame):
    """
    It takes a dataframe of classifications and returns a dataframe of annotations

    :param df: the dataframe containing the classifications
    :type df: pd.DataFrame
    :return: A dataframe with the following columns:
            classification_id, x, y, w, h, label, https_location, filename, subject_type, subject_ids,
            frame_number, user_name, movie_id
    """

    # Create an empty list
    rows_list = []

    # Loop through each classification submitted by the users and flatten them
    for index, row in df.iterrows():
        # Load annotations as json format
        annotations = json.loads(row["annotations"])

        # Select the information from all the labelled animals (e.g. task = T0)
        for ann_i in annotations:
            if ann_i["task"] == "T0":
                if ann_i["value"] == []:
                    # Specify the frame was classified as empty
                    choice_i = {
                        "classification_id": row["classification_id"],
                        "x": None,
                        "y": None,
                        "w": None,
                        "h": None,
                        "label": "empty",
                    }
                    rows_list.append(choice_i)

                else:
                    # Select each species annotated and flatten the relevant answers
                    for i in ann_i["value"]:
                        choice_i = {
                            "classification_id": row["classification_id"],
                            "x": int(i["x"]) if "x" in i else None,
                            "y": int(i["y"]) if "y" in i else None,
                            "w": int(i["width"]) if "width" in i else None,
                            "h": int(i["height"]) if "height" in i else None,
                            "label": str(i["tool_label"])
                            if "tool_label" in i
                            else None,
                        }
                        rows_list.append(choice_i)

    # Create a data frame with annotations as rows
    flat_annot_df = pd.DataFrame(
        rows_list, columns=["classification_id", "x", "y", "w", "h", "label"]
    )

    # Add other classification information to the flatten classifications
    annot_df = pd.merge(
        flat_annot_df,
        df,
        how="left",
        on="classification_id",
    )

    # Select only relevant columns
    annot_df = annot_df[
        [
            "classification_id",
            "x",
            "y",
            "w",
            "h",
            "label",
            "https_location",
            "filename",
            "subject_type",
            "subject_ids",
            "frame_number",
            "user_name",
            "movie_id",
            "workflow_version",
            "workflow_name",
            "workflow_id",
        ]
    ]

    return pd.DataFrame(annot_df)


def draw_annotations_in_frame(im: PILImage.Image, class_df_subject: pd.DataFrame):
    """
    > The function takes an image and a dataframe of annotations and returns the image with the
    annotations drawn on it

    :param im: the image object of type PILImage
    :param class_df_subject: a dataframe containing the annotations for a single subject
    :return: The image with the annotations
    """
    # Calculate image size
    dw, dh = im._size

    # Draw rectangles of each annotation
    img1 = ImageDraw.Draw(im)

    # Merge annotation info into a tuple
    class_df_subject["vals"] = class_df_subject[["x", "y", "w", "h"]].values.tolist()

    for index, row in class_df_subject.iterrows():
        # Specify the vals object
        vals = row.vals

        # Adjust annotantions to image size
        vals_adjusted = tuple(
            [
                int(vals[0]),
                int(vals[1]),
                int((vals[0] + vals[2])),
                int((vals[1] + vals[3])),
            ]
        )

        # Draw annotation
        img1.rectangle(vals_adjusted, width=2)

    return im


def view_subject(subject_id: int, class_df: pd.DataFrame, subject_type: str):
    """
    It takes a subject id, a dataframe containing the annotations for that subject, and the type of
    subject (clip or frame) and returns an HTML object that can be displayed in a notebook

    :param subject_id: The subject ID of the subject you want to view
    :type subject_id: int
    :param class_df: The dataframe containing the annotations for the class of interest
    :type class_df: pd.DataFrame
    :param subject_type: The type of subject you want to view. This can be either "clip" or "frame"
    :type subject_type: str
    """
    if subject_id in class_df.subject_ids.tolist():
        # Select the subject of interest
        class_df_subject = class_df[class_df.subject_ids == subject_id].reset_index(
            drop=True
        )

        # Get the location of the subject
        subject_location = class_df_subject["https_location"].unique()[0]

    else:
        raise Exception("The reference data does not contain media for this subject.")

    if len(subject_location) == 0:
        raise Exception("Subject not found in provided annotations")

    # Get the HTML code to show the selected subject
    if subject_type == "clip":
        html_code = f"""
        <html>
        <div style="display: flex; justify-content: space-around">
        <div>
          <video width=500 controls>
          <source src={subject_location} type="video/mp4">
        </video>
        </div>
        <div>{class_df_subject[['label','first_seen','how_many']].value_counts().sort_values(ascending=False).to_frame().to_html()}</div>
        </div>
        </html>"""

    elif subject_type == "frame":
        # Read image
        response = requests.get(subject_location)
        im = PILImage.open(BytesIO(response.content))

        # if label is not empty draw rectangles
        if class_df_subject.label.unique()[0] != "empty":
            # Create a temporary image with the annotations drawn on it
            im = draw_annotations_in_frame(im, class_df_subject)

        # Remove previous temp image if exist
        if os.access(".", os.W_OK):
            temp_image_path = "temp.jpg"
        else:
            # Specify volume allocated by SNIC
            snic_path = "/mimer/NOBACKUP/groups/snic2021-6-9"
            temp_image_path = f"{snic_path}/tmp_dir/temp.jpg"

        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        # Save the new image
        im.save(temp_image_path)

        # Load image data (used to enable viewing in Colab)
        img = open(temp_image_path, "rb").read()
        data_url = "data:image/jpeg;base64," + b64encode(img).decode()

        html_code = f"""
        <html>
        <div style="display: flex; justify-content: space-around">
        <div>
          <img src={data_url} type="image/jpeg" width=500>
        </img>
        </div>
        <div>{class_df_subject[['label','colour']].value_counts().sort_values(ascending=False).to_frame().to_html()}</div>
        </div>
        </html>"""
    else:
        Exception("Subject type not supported.")
    return HTML(html_code)


def launch_viewer(class_df: pd.DataFrame, subject_type: str):
    """
    > This function takes a dataframe of classifications and a subject type (frame or video) and
    displays a dropdown menu of subjects of that type. When a subject is selected, it displays the
    subject and the classifications for that subject

    :param class_df: The dataframe containing the classifications
    :type class_df: pd.DataFrame
    :param subject_type: The type of subject you want to view. This can be either "frame" or "video"
    :type subject_type: str
    """

    # If subject is frame assign a color to each label
    if subject_type == "frame":
        # Create a list of unique labels
        list_labels = class_df.label.unique().tolist()

        # Generate a list of random colors for each label
        random_color_list = []
        for index, item in enumerate(list_labels):
            random_color_list = random_color_list + [
                "#" + "".join([random.choice("ABCDEF0123456789") for i in range(6)])
            ]

        # Add a column with the color for each label
        class_df["colour"] = class_df.apply(
            lambda row: random_color_list[list_labels.index(row.label)], axis=1
        )

    # Select the subject
    options = tuple(
        class_df[class_df["subject_type"] == subject_type]["subject_ids"]
        .apply(int)
        .apply(str)
        .unique()
    )
    subject_widget = widgets.Dropdown(
        options=options,
        description="Subject id:",
        ensure_option=True,
        disabled=False,
    )

    main_out = widgets.Output()
    display(subject_widget, main_out)

    # Display the subject and classifications on change
    def on_change(change):
        with main_out:
            a = view_subject(int(change["new"]), class_df, subject_type)
            clear_output()
            display(a)

    subject_widget.observe(on_change, names="value")


def explore_classifications_per_subject(class_df: pd.DataFrame, subject_type: str):
    """
    > This function takes a dataframe of classifications and a subject type (clip or frame) and displays
    the classifications for a given subject

    :param class_df: the dataframe of classifications
    :type class_df: pd.DataFrame
    :param subject_type: "clip" or "frame"
    """

    # Select the subject
    subject_widget = widgets.Dropdown(
        options=tuple(class_df.subject_ids.apply(int).apply(str).unique()),
        description="Subject id:",
        ensure_option=True,
        disabled=False,
    )

    main_out = widgets.Output()
    display(subject_widget, main_out)

    # Display the subject and classifications on change
    def on_change(change):
        with main_out:
            a = class_df[class_df.subject_ids == int(change["new"])]
            if subject_type == "clip":
                a = a[
                    [
                        "classification_id",
                        "user_name",
                        "label",
                        "how_many",
                        "first_seen",
                    ]
                ]
            else:
                a = a[
                    [
                        "x",
                        "y",
                        "w",
                        "h",
                        "label",
                        "https_location",
                        "subject_ids",
                        "frame_number",
                        "movie_id",
                    ]
                ]
            clear_output()
            display(a)

    subject_widget.observe(on_change, names="value")


def encode_image(filepath):
    """
    It takes a filepath to an image, opens the image, reads the bytes, encodes the bytes as base64, and
    returns the encoded string

    :param filepath: The path to the image file
    :return: the base64 encoding of the image.
    """
    with open(filepath, "rb") as f:
        image_bytes = f.read()
    encoded = str(b64encode(image_bytes), "utf-8")
    return "data:image/jpg;base64," + encoded


def get_annotations_viewer(data_path: str, species_list: list):
    """
    It takes a path to a folder containing images and annotations, and a list of species names, and
    returns a widget that allows you to view the images and their annotations, and to edit the
    annotations

    :param data_path: the path to the data folder
    :type data_path: str
    :param species_list: a list of species names
    :type species_list: list
    :return: A VBox widget containing a progress bar and a BBoxWidget.
    """
    image_path = os.path.join(data_path, "images")
    annot_path = os.path.join(data_path, "labels")

    images = sorted(
        [
            f
            for f in os.listdir(image_path)
            if os.path.isfile(os.path.join(image_path, f))
        ]
    )
    annotations = sorted(
        [
            f
            for f in os.listdir(annot_path)
            if os.path.isfile(os.path.join(annot_path, f))
        ]
    )

    if any([len(images), len(annotations)]) == 0:
        logging.error("No annotations to display")
        return None

    # a progress bar to show how far we got
    w_progress = widgets.IntProgress(value=0, max=len(images), description="Progress")
    # the bbox widget
    image = os.path.join(image_path, images[0])
    width, height = imagesize.get(image)
    label_file = annotations[w_progress.value]
    bboxes = []
    labels = []
    with open(os.path.join(annot_path, label_file), "r") as f:
        for line in f:
            s = line.split(" ")
            labels.append(s[0])

            left = (float(s[1]) - (float(s[3]) / 2)) * width
            top = (float(s[2]) - (float(s[4]) / 2)) * height

            bboxes.append(
                {
                    "x": left,
                    "y": top,
                    "width": float(s[3]) * width,
                    "height": float(s[4]) * height,
                    "label": species_list[int(s[0])],
                }
            )
    w_bbox = widgets.BBoxWidget(image=encode_image(image), classes=species_list)

    # here we assign an empty list to bboxes but
    # we could also run a detection model on the file
    # and use its output for creating inital bboxes
    w_bbox.bboxes = bboxes

    # combine widgets into a container
    w_container = widgets.VBox(
        [
            w_progress,
            w_bbox,
        ]
    )

    def on_button_clicked(b):
        w_progress.value = 0
        image = os.path.join(image_path, images[0])
        width, height = imagesize.get(image)
        label_file = annotations[w_progress.value]
        bboxes = []
        labels = []
        with open(os.path.join(annot_path, label_file), "r") as f:
            for line in f:
                s = line.split(" ")
                labels.append(s[0])

                left = (float(s[1]) - (float(s[3]) / 2)) * width
                top = (float(s[2]) - (float(s[4]) / 2)) * height

                bboxes.append(
                    {
                        "x": left,
                        "y": top,
                        "width": float(s[3]) * width,
                        "height": float(s[4]) * height,
                        "label": species_list[int(s[0])],
                    }
                )
        w_bbox.image = encode_image(image)

        # here we assign an empty list to bboxes but
        # we could also run a detection model on the file
        # and use its output for creating inital bboxes
        w_bbox.bboxes = bboxes
        w_container.children = tuple(list(w_container.children[1:]))
        b.close()

    # when Skip button is pressed we move on to the next file
    def on_skip():
        w_progress.value += 1
        if w_progress.value == len(annotations):
            button = widgets.Button(
                description="Click to restart.",
                disabled=False,
                display="flex",
                flex_flow="column",
                align_items="stretch",
            )
            if isinstance(w_container.children[0], widgets.Button):
                w_container.children = tuple(list(w_container.children[1:]))
            w_container.children = tuple([button] + list(w_container.children))
            button.on_click(on_button_clicked)

        # open new image in the widget
        else:
            image_file = images[w_progress.value]
            image_p = os.path.join(image_path, image_file)
            width, height = imagesize.get(image_p)
            w_bbox.image = encode_image(image_p)
            label_file = annotations[w_progress.value]
            bboxes = []
            with open(os.path.join(annot_path, label_file), "r") as f:
                for line in f:
                    s = line.split(" ")
                    left = (float(s[1]) - (float(s[3]) / 2)) * width
                    top = (float(s[2]) - (float(s[4]) / 2)) * height
                    bboxes.append(
                        {
                            "x": left,
                            "y": top,
                            "width": float(s[3]) * width,
                            "height": float(s[4]) * height,
                            "label": species_list[int(s[0])],
                        }
                    )

            # here we assign an empty list to bboxes but
            # we could also run a detection model on the file
            # and use its output for creating initial bboxes
            w_bbox.bboxes = bboxes

    w_bbox.on_skip(on_skip)

    # when Submit button is pressed we save current annotations
    # and then move on to the next file
    def on_submit():
        image_file = images[w_progress.value]
        width, height = imagesize.get(os.path.join(image_path, image_file))
        # save annotations for current image
        open(os.path.join(annot_path, label_file), "w").write(
            "\n".join(
                [
                    "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                        species_list.index(
                            i["label"]
                        ),  # single class vs multiple classes
                        min((i["x"] + i["width"] / 2) / width, 1.0),
                        min((i["y"] + i["height"] / 2) / height, 1.0),
                        min(i["width"] / width, 1.0),
                        min(i["height"] / height, 1.0),
                    )
                    for i in w_bbox.bboxes
                ]
            )
        )
        # move on to the next file
        on_skip()

    w_bbox.on_submit(on_submit)

    return w_container


def get_workflow_labels(
    workflow_df: pd.DataFrame, workflow_id: int, workflow_version: int
):
    """
    > This function takes a df of workflows of interest and retrieves the labels and common names of the choices cit scientists have in a survey task in Zooniverse.
    the function is a modified version of the 'get_workflow_info' function by @lcjohnso
    https://github.com/zooniverse/Data-digging/blob/6e9dc5db6f6125316616c4b04ae5fc4223826a25/scripts_GeneralPython/get_workflow_info.pybiological observations classified by citizen scientists, biologists or ML algorithms and returns a df of species occurrences to publish in GBIF/OBIS.
    :param workflow_df: df of the workflows of the Zooniverse project of interest,
    :param workflow_id: integer of the workflow id of interest,
    :param workflow_version: integer of the workflow version of interest.
    :return: a df with the common name and label of the annotations for the workflow.
    """
    # initialize the output
    workflow_info = {}

    # parse the tasks column as a json so we can work with it (it just loads as a string)
    workflow_df["tasks_json"] = [json.loads(q) for q in workflow_df["tasks"]]
    workflow_df["strings_json"] = [json.loads(q) for q in workflow_df["strings"]]

    # identify the row of the workflow dataframe we want to extract
    is_theworkflow = (workflow_df["workflow_id"] == workflow_id) & (
        workflow_df["version"] == workflow_version
    )

    # extract it
    theworkflow = workflow_df[is_theworkflow]

    # pandas is a little weird about accessing stuff sometimes
    # we should only have 1 row in theworkflow but the row index will be retained
    # from the full workflow_df, so we need to figure out what it is
    i_wf = theworkflow.index[0]

    # extract the tasks as a json
    tasks = theworkflow["tasks_json"][i_wf]
    strings = theworkflow["strings_json"][i_wf]

    workflow_info = tasks.copy()

    tasknames = workflow_info.keys()
    workflow_info["tasknames"] = tasknames

    # now that we've extracted the actual task names, add the first task
    workflow_info["first_task"] = theworkflow["first_task"].values[0]

    # now join workflow structure to workflow label content for each task

    for task in tasknames:
        # Create an empty dictionary to host the dfs of interest
        label_common_name_dict = {"commonName": [], "label": []}

        # Create an empty dictionary to host the dfs of interest
        label_common_name_dict = {"commonName": [], "label": []}
        for i_c, choice in enumerate(workflow_info[task]["choices"].keys()):
            c_label = strings[workflow_info[task]["choices"][choice]["label"]]
            label_common_name_dict["commonName"].append(choice)
            label_common_name_dict["label"].append(c_label)

        if task == "T0":
            break

    return pd.DataFrame.from_dict(label_common_name_dict)


def choose_test_prop():
    """
    > The function `choose_test_prop()` creates a slider widget that allows the user to choose the
    proportion of the data to be used for testing
    :return: A widget object
    """

    w = widgets.FloatSlider(
        value=0.2,
        min=0.0,
        max=1.0,
        step=0.1,
        description="Test proportion:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
    )

    display(w)
    return w


def choose_eval_params():
    """
    It creates one slider for confidence threshold
    :return: the value of the slider.
    """

    z1 = widgets.FloatSlider(
        value=0.5,
        min=0.0,
        max=1.0,
        step=0.1,
        description="Confidence threshold:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
    )

    display(z1)
    return z1


def choose_train_params(model_type: str):
    """
    It creates two sliders, one for batch size, one for epochs
    :return: the values of the sliders.
    """
    v = widgets.FloatLogSlider(
        value=1,
        base=2,
        min=0,  # max exponent of base
        max=10,  # min exponent of base
        step=1,  # exponent step
        description="Batch size:",
        readout=True,
        readout_format="d",
    )

    z = widgets.IntSlider(
        value=1,
        min=0,
        max=1000,
        step=10,
        description="Epochs:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
    )

    h = widgets.IntText(description="Height:")
    w = widgets.IntText(description="Width:")
    s = widgets.IntText(description="Image size:")

    def on_value_change(change):
        height = h.value
        width = w.value
        return [height, width]

    h.observe(on_value_change, names="value")
    w.observe(on_value_change, names="value")
    s.observe(on_value_change, names="value")

    if model_type == 1:
        box = widgets.HBox([v, z, h, w])
        display(box)
        return v, z, h, w
    elif model_type == 2:
        box = widgets.HBox([v, z, s])
        display(box)
        return v, z, s, None
    else:
        logging.warning("Model in experimental stage.")
        box = widgets.HBox([v, z])
        display(box)
        return v, z, None, None


def choose_experiment_name():
    """
    It creates a text box that allows you to enter a name for your experiment
    :return: The text box widget.
    """
    exp_name = widgets.Text(
        value="exp_name",
        placeholder="Choose an experiment name",
        description="Experiment name:",
        disabled=False,
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
    )
    display(exp_name)
    return exp_name


def choose_entity():
    """
    It creates a text box that allows you to enter your username or teamname of WandB
    :return: The text box widget.
    """
    entity = widgets.Text(
        value="koster",
        placeholder="Give your user or team name",
        description="User or Team name:",
        disabled=False,
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
    )
    display(entity)
    return entity


def choose_model_type():
    """
    It creates a dropdown box that allows you to choose a model type
    :return: The dropdown box widget.
    """
    model_type = widgets.Dropdown(
        value=None,
        description="Required model type:",
        options=[
            (
                "Object Detection (e.g. identifying individuals in an image using rectangles)",
                1,
            ),
            (
                "Image Classification (e.g. assign a class or label to an entire image)",
                2,
            ),
            (
                "Instance Segmentation (e.g. fit a suitable mask on top of identified objects)",
                3,
            ),
            ("Custom model (currently only Faster RCNN)", 4),
        ],
        disabled=False,
        display="flex",
        flex_flow="column",
        align_items="stretch",
        layout={"width": "max-content"},
        style={"description_width": "initial"},
    )
    display(model_type)
    return model_type


def choose_conf():
    w = widgets.FloatSlider(
        value=0.5,
        min=0,
        max=1.0,
        step=0.1,
        description="Confidence threshold:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
    )
    display(w)
    return w


def choose_text(name: str):
    text_widget = widgets.Text(
        description=f"Please enter a suitable {name} ",
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
    )
    display(text_widget)
    return text_widget


class WidgetMaker(widgets.VBox):
    def __init__(self):
        """
        The function creates a widget that allows the user to select which workflows to run

        :param workflows_df: the dataframe of workflows
        """
        self.widget_count = widgets.IntText(
            description="Number of authors:",
            display="flex",
            flex_flow="column",
            align_items="stretch",
            style={"description_width": "initial"},
        )

        self.bool_widget_holder = widgets.HBox(
            layout=widgets.Layout(
                width="100%", display="inline-flex", flex_flow="row wrap"
            )
        )
        children = [
            self.widget_count,
            self.bool_widget_holder,
        ]
        self.widget_count.observe(self._add_bool_widgets, names=["value"])
        super().__init__(children=children)

    def _add_bool_widgets(self, widg):
        num_bools = widg["new"]
        new_widgets = []
        for _ in range(num_bools):
            new_widget = widgets.Text(
                display="flex",
                flex_flow="column",
                align_items="stretch",
                style={"description_width": "initial"},
            ), widgets.Text(
                display="flex",
                flex_flow="column",
                align_items="stretch",
                style={"description_width": "initial"},
            )
            new_widget[0].description = "Author Name: " + f" #{_}"
            new_widget[1].description = "Organisation: " + f" #{_}"
            new_widgets.extend(new_widget)
        self.bool_widget_holder.children = tuple(new_widgets)

    @property
    def checks(self):
        return {w.description: w.value for w in self.bool_widget_holder.children}

    @property
    def author_dict(self):
        init_dict = {w.description: w.value for w in self.bool_widget_holder.children}
        names, organisations = [], []
        for i in range(0, len(init_dict), 2):
            names.append(list(init_dict.values())[i])
            organisations.append(list(init_dict.values())[i + 1])
        return {n: org for n, org in zip(names, organisations)}
