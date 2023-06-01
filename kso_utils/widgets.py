# base imports
import logging
import os
import math
import random
import subprocess
import datetime
import pandas as pd
import numpy as np
import cv2

# widget imports
import ipysheet
import folium
import ipywidgets as widgets
from ipyfilechooser import FileChooser
from IPython.display import HTML, display, clear_output
from ipywidgets import interactive, Layout
from folium.plugins import MiniMap
from pathlib import Path
import asyncio

# util imports
from kso_utils.project_utils import Project
import kso_utils.movie_utils as movie_utils
from kso_utils.db_utils import create_connection
import kso_utils.tutorials_utils as t_utils

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def choose_folder(start_path: str = ".", folder_type: str = ""):
    # Specify the output folder
    fc = FileChooser(start_path)
    fc.title = f"Choose location of {folder_type}"
    display(fc)
    return fc


def choose_footage(
    project: Project, db_info_dict: dict, start_path: str = ".", folder_type: str = ""
):
    if project.server == "AWS":
        available_movies_df = movie_utils.retrieve_movie_info_from_server(
            project=project, db_info_dict=db_info_dict
        )
        movie_dict = {
            name: movie_utils.get_movie_path(f_path, db_info_dict, project)
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


def select_random_clips(project: Project, movie_i: str):
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
    conn = create_connection(project.db_path)

    # Query info about the movie of interest
    movie_df = pd.read_sql_query(
        f"SELECT filename, duration, sampling_start, sampling_end FROM movies WHERE filename='{movie_i}'",
        conn,
    )

    # Select n number of clips at random
    def n_random_clips(clip_length, n_clips):
        # Create a list of starting points for n number of clips
        duration_movie = math.floor(movie_df["duration"].values[0])
        starting_clips = random.sample(range(0, duration_movie, clip_length), n_clips)

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
        clip_length=select_clip_length(),
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


def select_clip_n_len(project: Project, movie_i: str):
    """
    This function allows the user to select the length of the clips to upload to the database

    :param movie_i: the name of the movie you want to upload
    :param db_info_dict: a dictionary containing the path to the database and the name of the database
    :return: The number of clips to upload
    """

    # Create connection to db
    conn = create_connection(project.db_path)

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
        clip_length=select_clip_length(),
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


def choose_species(project: Project):
    """
    This function generates a widget to select the species of interest
    :param db_info_dict: a dictionary containing the path to the database
    :type db_info_dict: dict
    """
    # Create connection to db
    conn = create_connection(project.db_path)

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


def choose_classes(db_path: str = "koster_lab.db"):
    """
    It creates a dropdown menu of all the species in the database, and returns the species that you
    select

    :param db_path: The path to the database, defaults to koster_lab.db
    :type db_path: str (optional)
    :return: A widget object
    """
    conn = create_connection(db_path)
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


def map_sites(project: Project, db_info_dict: dict):
    """
    > This function takes a dictionary of database information and a project object as input, and
    returns a map of the sites in the database

    :param db_info_dict: a dictionary containing the information needed to connect to the database
    :type db_info_dict: dict
    :param project: The project object
    :return: A map with all the sites plotted on it.
    """
    if project.server in ["SNIC", "LOCAL"]:
        # Set initial location to Gothenburg
        init_location = [57.708870, 11.974560]

    else:
        # Set initial location to Taranaki
        init_location = [-39.296109, 174.063916]

    # Create the initial kso map
    kso_map = folium.Map(location=init_location, width=900, height=600)

    # Read the csv file with site information
    sites_df = pd.read_csv(db_info_dict["local_sites_csv"])

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
    kso_map = kso_map.add_child(MiniMap())

    # Return the map
    return kso_map


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


def gpu_select():
    """
    If the user selects "No GPU", then the function will return a boolean value of False and use the CPU. If the user
    selects "GPU" function will return a boolean value of True
    :return: The gpu_available variable is being returned.
    """

    def gpu_output(gpu_option):
        if gpu_option == "No GPU":
            gpu_available = False
            return gpu_available

        if gpu_option == "Colab, local or server GPU available":
            gpu_available = True
            return gpu_available

    # Select the gpu availability
    gpu_output_interact = interactive(
        gpu_output,
        gpu_option=widgets.RadioButtons(
            options=["No GPU", "Colab, local or server GPU available"],
            value="No GPU",
            description="Select GPU availability:",
            disabled=False,
        ),
    )

    display(gpu_output_interact)
    return gpu_output_interact


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


class clip_modification_widget(widgets.VBox):
    def __init__(self, gpu_available):
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
        self.gpu_available = gpu_available
        super().__init__(children=children)

    def _add_bool_widgets(self, widg):
        num_bools = widg["new"]
        new_widgets = []
        for _ in range(num_bools):
            new_widget = select_modification(self.gpu_available)
            for wdgt in [new_widget]:
                wdgt.description = wdgt.description + f" #{_}"
            new_widgets.extend([new_widget])
        self.bool_widget_holder.children = tuple(new_widgets)

    @property
    def checks(self):
        return {w.description: w.value for w in self.bool_widget_holder.children}


def select_modification(gpu_available):
    """
    This function creates a dropdown widget that allows the user to select a clip modification
    :return: A widget that allows the user to select a clip modification.
    """
    # Widget to select the clip modification
    if gpu_available == False:
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
    
        if gpu_available == True:
            clip_modifications = {
                "Color_correction": {
                    "filter": "curves=r='0/0 0.396/0.67 1/1':g='0/0 0.525/0.451 1/1':b='0/0 0.459/0.517 1/1'"
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
                "Blur_sensitive_info": { ## NOT POSSIBLE YET
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


def select_survey(db_info_dict: dict):
    """
    This function allows the user to select an existing survey from a dropdown menu or create a new
    survey by filling out a series of widgets

    :param db_info_dict: a dictionary with the following keys:
    :type db_info_dict: dict
    :return: A widget with a dropdown menu with two options: 'Existing' and 'New survey'.
    """
    # Load the csv with surveys information
    surveys_df = pd.read_csv(db_info_dict["local_surveys_csv"])

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
            choices_df = pd.read_csv(db_info_dict["local_choices_csv"])

            # Save the new survey responses into a dict
            survey_info = {
                # Write the name of the encoder
                "EncoderName": record_encoder(),
                # Select the start date of the survey
                "SurveyStartDate": select_SurveyStartDate(),
                # Write the name of the survey
                "SurveyName": write_SurveyName(),
                # Select the DOC office
                "OfficeName": select_OfficeName(
                    choices_df.OfficeName.dropna().unique().tolist()
                ),
                # Write the name of the contractor
                "ContractorName": write_ContractorName(),
                # Write the number of the contractor
                "ContractNumber": write_ContractNumber(),
                # Write the link to the contract
                "LinkToContract": write_LinkToContract(),
                # Record the name of the survey leader
                "SurveyLeaderName": write_SurveyLeaderName(),
                # Select the name of the linked Marine Reserve
                "LinkToMarineReserve": select_LinkToMarineReserve(
                    choices_df.MarineReserve.dropna().unique().tolist()
                ),
                # Specify if survey is single species
                "FishMultiSpecies": select_FishMultiSpecies(),
                # Specify how the survey was stratified
                "StratifiedBy": select_StratifiedBy(
                    choices_df.StratifiedBy.dropna().unique().tolist()
                ),
                # Select if survey is part of long term monitoring
                "IsLongTermMonitoring": select_IsLongTermMonitoring(),
                # Specify the site selection of the survey
                "SiteSelectionDesign": select_SiteSelectionDesign(
                    choices_df.SiteSelection.dropna().unique().tolist()
                ),
                # Specify the unit selection of the survey
                "UnitSelectionDesign": select_UnitSelectionDesign(
                    choices_df.UnitSelection.dropna().unique().tolist()
                ),
                # Select the type of right holder of the survey
                "RightsHolder": select_RightsHolder(
                    choices_df.RightsHolder.dropna().unique().tolist()
                ),
                # Write who can access the videos/resources
                "AccessRights": select_AccessRights(),
                # Write a description of the survey design and objectives
                "SurveyVerbatim": write_SurveyVerbatim(),
                # Select the type of BUV
                "BUVType": select_BUVType(
                    choices_df.BUVType.dropna().unique().tolist()
                ),
                # Write the link to the pictures
                "LinkToPicture": write_LinkToPicture(),
                # Write the name of the vessel
                "Vessel": write_Vessel(),
                # Write the link to the fieldsheets
                "LinkToFieldSheets": write_LinkToFieldSheets(),
                # Write the link to LinkReport01
                "LinkReport01": write_LinkReport01(),
                # Write the link to LinkReport02
                "LinkReport02": write_LinkReport02(),
                # Write the link to LinkReport03
                "LinkReport03": write_LinkReport03(),
                # Write the link to LinkReport04
                "LinkReport04": write_LinkReport04(),
                # Write the link to LinkToOriginalData
                "LinkToOriginalData": write_LinkToOriginalData(),
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


def record_deployment_info(db_info_dict: dict, deployment_filenames: list):
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
        movies_df = pd.read_csv(db_info_dict["local_movies_csv"])

        # Load the csv with with sites and survey choices
        choices_df = pd.read_csv(db_info_dict["local_choices_csv"])

        deployment_info = {
            # Select the start of the sampling
            "SamplingStart": select_SamplingStart(duration),
            # Select the end of the sampling
            "SamplingEnd": select_SamplingEnd(duration),
            # Specify if deployment is bad
            "IsBadDeployment": select_IsBadDeployment(),
            # Write the number of the replicate within the site
            "ReplicateWithinSite": write_ReplicateWithinSite(),
            # Select the person who recorded this deployment
            "RecordedBy": select_RecordedBy(movies_df.RecordedBy.unique()),
            # Select depth stratum of the deployment
            "DepthStrata": select_DepthStrata(),
            # Select the depth of the deployment
            "Depth": select_Depth(),
            # Select the underwater visibility
            "UnderwaterVisibility": select_UnderwaterVisibility(
                choices_df.UnderwaterVisibility.dropna().unique().tolist()
            ),
            # Select the time in
            "TimeIn": deployment_TimeIn(),
            # Select the time out
            "TimeOut": deployment_TimeOut(),
            # Add any comment related to the deployment
            "NotesDeployment": write_NotesDeployment(),
            # Select the theoretical duration of the deployment
            "DeploymentDurationMinutes": select_DeploymentDurationMinutes(),
            # Write the type of habitat
            "Habitat": write_Habitat(),
            # Write the number of NZMHCS_Abiotic
            "NZMHCS_Abiotic": write_NZMHCS_Abiotic(),
            # Write the number of NZMHCS_Biotic
            "NZMHCS_Biotic": write_NZMHCS_Biotic(),
            # Select the level of the tide
            "TideLevel": select_TideLevel(
                choices_df.TideLevel.dropna().unique().tolist()
            ),
            # Describe the weather of the deployment
            "Weather": write_Weather(),
            # Select the model of the camera used
            "CameraModel": select_CameraModel(
                choices_df.CameraModel.dropna().unique().tolist()
            ),
            # Write the camera lens and settings used
            "LensModel": write_LensModel(),
            # Specify the type of bait used
            "BaitSpecies": write_BaitSpecies(),
            # Specify the amount of bait used
            "BaitAmount": select_BaitAmount(),
        }

        return deployment_info


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
        # Retrieve the column name of the id of interest (Sites, movies,..)
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


def extract_custom_frames(
    input_path,
    output_dir,
    skip_start=None,
    skip_end=None,
    num_frames=None,
    frame_skip=None,
):
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
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_start = int(skip_start * fps)
    skip_end = int(skip_end * fps)

    frame_start = 0 if skip_start is None else skip_start
    frame_end = num_frames_total if skip_end is None else num_frames_total - skip_end

    # Determine which frames to extract based on the input parameters
    if num_frames is not None:
        frames_to_extract = random.sample(range(frame_start, frame_end), num_frames)
    elif frame_skip is not None:
        frames_to_extract = range(frame_start, frame_end, frame_skip)
    else:
        frames_to_extract = range(frame_end)

    output_files, input_movies = [], []

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

            # Add movie filename to list
            input_movies.append(Path(input_path).name)

    # Release the video capture object
    cap.release()

    return pd.DataFrame(
        np.column_stack([input_movies, output_files, frames_to_extract]),
        columns=["movie_filename", "frame_path", "frame_number"],
    )


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


def confirm_survey(db_info_dict: dict, survey_i):
    """
    It takes the survey information and checks if it's a new survey or an existing one. If it's a new
    survey, it saves the information in the survey csv file. If it's an existing survey, it prints the
    information for the pre-existing survey

    :param survey_i: the survey widget
    :param db_info_dict: a dictionary with the following keys:
    """

    from kso_utils.server_utils import upload_file_to_s3

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
        choices_df = pd.read_csv(db_info_dict["local_choices_csv"])

        # Get prepopulated fields for the survey
        new_survey_row["OfficeContact"] = choices_df[
            choices_df["OfficeName"] == new_survey_row.OfficeName.values[0]
        ]["OfficeContact"].values[0]
        new_survey_row[["SurveyLocation", "Region"]] = choices_df[
            choices_df["MarineReserve"] == new_survey_row.LinkToMarineReserve.values[0]
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
        logging.info("The details of the new survey are:")
        for ind in new_survey_row.T.index:
            logging.info(ind, "-->", new_survey_row.T[0][ind])

        # Save changes in survey csv locally and in the server

        async def f(new_survey_row):
            x = await wait_for_change(
                correct_button, wrong_button
            )  # <---- Pass both buttons into the function
            if (
                x == "Yes, details are correct"
            ):  # <--- use if statement to trigger different events for the two buttons
                # Load the csv with sites information
                surveys_df = pd.read_csv(db_info_dict["local_surveys_csv"])

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
                if new_survey_row.SurveyID.unique()[0] in surveys_df.SurveyID.unique():
                    logging.error(
                        f"The survey {new_survey_row.SurveyID.unique()[0]} already exists in the database."
                    )
                    raise

                logging.info("Updating the new survey information.")

                # Add the new row to the choices df
                surveys_df = surveys_df.append(new_survey_row, ignore_index=True)

                # Save the updated df locally
                surveys_df.to_csv(db_info_dict["local_surveys_csv"], index=False)

                # Save the updated df in the server
                upload_file_to_s3(
                    db_info_dict["client"],
                    bucket=db_info_dict["bucket"],
                    key=db_info_dict["server_surveys_csv"],
                    filename=db_info_dict["local_surveys_csv"].__str__(),
                )

                logging.info("Survey information updated!")

            else:
                logging.info("Come back when the data is tidy!")

    # If existing survey print the info for the pre-existing survey
    else:
        # Load the csv with surveys information
        surveys_df = pd.read_csv(db_info_dict["local_surveys_csv"])

        # Select the specific survey info
        new_survey_row = surveys_df[
            surveys_df["SurveyName"] == survey_i.result.value
        ].reset_index(drop=True)

        logging.info("The details of the selected survey are:")
        for ind in new_survey_row.T.index:
            logging.info(ind, "-->", new_survey_row.T[0][ind])

        async def f(new_survey_row):
            x = await wait_for_change(
                correct_button, wrong_button
            )  # <---- Pass both buttons into the function
            if (
                x == "Yes, details are correct"
            ):  # <--- use if statement to trigger different events for the two buttons
                logging.info("Great, you can start uploading the movies.")

            else:
                logging.info("Come back when the data is tidy!")

    logging.info("")
    logging.info("")
    logging.info("Are the survey details above correct?")
    display(
        widgets.HBox([correct_button, wrong_button])
    )  # <----Display both buttons in an HBox
    asyncio.create_task(f(new_survey_row))


def update_meta(
    project: Project,
    db_info_dict: dict,
    sheet_df: pd.DataFrame,
    df: pd.DataFrame,
    meta_name: str,
):
    """
    `update_meta` takes a new table, a meta name, and updates the local and server meta files

    :param sheet_df: The dataframe of the sheet you want to update
    :param meta_name: the name of the metadata file (e.g. "movies")
    """

    from kso_utils.db_utils import process_test_csv
    from kso_utils.server_utils import update_csv_server

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
    async def f(sheet_df, df, meta_name):
        x = await wait_for_change(
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
            csv_i, df_to_db = process_test_csv(
                project=project,
                local_csv=str(db_info_dict["local_" + meta_name + "_csv"]),
            )

            # Log changes locally
            t_utils.log_meta_changes(
                project=project,
                db_info_dict=db_info_dict,
                meta_key="local_" + meta_name + "_csv",
                new_sheet_df=sheet_df,
            )

            # Save the updated df locally
            df_orig.to_csv(db_info_dict["local_" + meta_name + "_csv"], index=False)
            logging.info("The local csv file has been updated")

            if project.server == "AWS":
                # Save the updated df in the server
                update_csv_server(
                    project=project,
                    db_info_dict=db_info_dict,
                    orig_csv="server_" + meta_name + "_csv",
                    updated_csv="local_" + meta_name + "_csv",
                )

        else:
            logging.info("Run this cell again when the changes are correct!")

    logging.info("")
    logging.info("Are the changes above correct?")
    display(
        widgets.HBox([confirm_button, deny_button])
    )  # <----Display both buttons in an HBox
    asyncio.create_task(f(sheet_df, df, meta_name))


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
