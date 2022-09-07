## Zooniverse utils

# base imports
import io
import getpass
import pandas as pd
import json
import logging
import numpy as np
from panoptes_client import (
    Project,
    Panoptes,
)
from ast import literal_eval

# util imports
from kso_utils.koster_utils import (
    process_koster_subjects,
    clean_duplicated_subjects,
    combine_annot_from_duplicates,
)
from kso_utils.spyfish_utils import process_spyfish_subjects
import kso_utils.db_utils as db_utils
import kso_utils.project_utils as project_utils

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def zoo_credentials():
    zoo_user = getpass.getpass("Enter your Zooniverse user")
    zoo_pass = getpass.getpass("Enter your Zooniverse password")
    return zoo_user, zoo_pass


class AuthenticationError(Exception):
    pass


# Function to authenticate to Zooniverse
def auth_session(username: str, password: str, project_n: int):
    """
    It connects to the Zooniverse with your username and password, and then returns the project number
    you specify

    :param username: Your Zooniverse username
    :param password: your Zooniverse password
    :param project_n: The project number of the project you want to download data from
    :return: The project number of the koster lab
    """

    # Connect to Zooniverse with your username and password
    auth = Panoptes.connect(username=username, password=password)

    if not auth.logged_in:
        raise AuthenticationError("Your credentials are invalid. Please try again.")

    # Specify the project number of the koster lab
    try:
        project = Project(int(float(project_n)))
        return project
    except Exception as e:
        logging.error(e)


# Function to retrieve information from Zooniverse
def retrieve_zoo_info(
    project: project_utils.Project, zoo_project: Project, zoo_info: str
):
    """
    This function retrieves the information of interest from Zooniverse and saves it as a pandas data
    frame

    :param project: the project object
    :param zoo_project: the Zooniverse project object
    :param zoo_info: a list of the info you want to retrieve from Zooniverse
    :type zoo_info: str
    :return: A dictionary of dataframes.
    """
    if hasattr(project, "info_df"):
        if project.info_df is not None:
            print(
                "Zooniverse info retrieved from cache, to force retrieval set project.info_df = None"
            )
            return project.info_df
    # Create an empty dictionary to host the dfs of interest
    info_df = {}

    for info_n in zoo_info:
        print("Retrieving", info_n, "from Zooniverse")

        # Get the information of interest from Zooniverse
        export = zoo_project.get_export(info_n)

        # Save the info as pandas data frame
        export_df = pd.read_csv(io.StringIO(export.content.decode("utf-8")))

        if len(export_df) > 0:

            # If KSO deal with duplicated subjects
            if project.Project_name == "Koster_Seafloor_Obs":

                # Clear duplicated subjects
                if info_n == "subjects":
                    export_df = clean_duplicated_subjects(export_df, project)

                # Combine classifications from duplicated subjects to unique subject id
                if info_n == "classifications":
                    export_df = combine_annot_from_duplicates(export_df, project)

        else:
            raise ValueError(
                "The export is empty. This may be due to a "
                "request time out, please try again in 1 minute."
            )

        # Ensure subject_ids match db format
        if info_n == "classifications":
            export_df["subject_ids"] = export_df["subject_ids"].astype(np.int64)

        # Add df to dictionary
        info_df[info_n] = export_df
        project.info_df = info_df
        print(info_n, "were retrieved successfully")

    return info_df


# Function to extract metadata from subjects
def extract_metadata(subj_df: pd.DataFrame):
    """
    > The function takes a dataframe with a column called `metadata` that contains a JSON string. It
    then flattens the JSON string into a dataframe and returns the original dataframe with the
    `metadata` column removed and the flattened dataframe

    :param subj_df: The dataframe containing the subject data
    :return: A tuple of two dataframes.
    """

    # Reset index of df
    subj_df = subj_df.reset_index(drop=True).reset_index()

    # Flatten the metadata information
    meta_df = pd.json_normalize(subj_df.metadata.apply(json.loads))

    # Drop metadata and index columns from original df
    subj_df = subj_df.drop(
        columns=[
            "metadata",
            "index",
        ]
    )

    return subj_df, meta_df


def populate_subjects(
    subjects: pd.DataFrame, project: project_utils.Project, db_path: str
):
    """
    Populate the subjects table with the subject metadata

    :param subjects: the subjects dataframe
    :param project_path: The path to the projects.csv file
    :param project_name: The name of the Zooniverse project
    :param db_path: the path to the database
    """

    project_name = project.Project_name
    server = project.server
    movie_folder = project.movie_folder

    # Check if the Zooniverse project is the KSO
    if project_name == "Koster_Seafloor_Obs":

        subjects = process_koster_subjects(subjects, db_path)

    else:

        # Extract metadata from uploaded subjects
        subjects_df, subjects_meta = extract_metadata(subjects)

        # Combine metadata info with the subjects df
        subjects = pd.concat([subjects_df, subjects_meta], axis=1)

        # Check if the Zooniverse project is the Spyfish
        if project_name == "Spyfish_Aotearoa":

            subjects = process_spyfish_subjects(subjects, db_path)

    # Set subject_id information as id
    subjects = subjects.rename(columns={"subject_id": "id"})

    # Extract the html location of the subjects
    subjects["https_location"] = subjects["locations"].apply(
        lambda x: literal_eval(x)["0"]
    )

    # Set movie_id column to None if no movies are linked to the subject
    if movie_folder == "None" and server in ["local", "SNIC"]:
        subjects["movie_id"] = None

    # Fix weird bug where Subject_type is used instead of subject_type for the column name for some clips
    subjects["subject_type"] = subjects[["subject_type", "Subject_type"]].apply(
        lambda x: x[1] if isinstance(x[1], str) else x[0], 1
    )

    # Set the columns in the right order
    subjects = subjects[
        [
            "id",
            "subject_type",
            "filename",
            "clip_start_time",
            "clip_end_time",
            "frame_exp_sp_id",
            "frame_number",
            "workflow_id",
            "subject_set_id",
            "classifications_count",
            "retired_at",
            "retirement_reason",
            "created_at",
            "https_location",
            "movie_id",
        ]
    ]

    # Ensure that subject_ids are not duplicated by workflow
    subjects = subjects.drop_duplicates(subset="id")

    # Test table validity
    db_utils.test_table(subjects, "subjects", keys=["id"])

    # Add values to subjects
    db_utils.add_to_table(db_path, "subjects", [tuple(i) for i in subjects.values], 15)

    ##### Print how many subjects are in the db
    # Create connection to db
    conn = db_utils.create_connection(db_path)

    # Query id and subject type from the subjects table
    subjects_df = pd.read_sql_query("SELECT id, subject_type FROM subjects", conn)
    frame_subjs = subjects_df[subjects_df["subject_type"] == "frame"].shape[0]
    clip_subjs = subjects_df[subjects_df["subject_type"] == "clip"].shape[0]

    print(
        "The database has a total of",
        frame_subjs,
        "frame subjects and",
        clip_subjs,
        "clip subjects have been updated",
    )


# Relevant for ML and upload frames tutorials
def populate_agg_annotations(
    annotations: pd.DataFrame, subj_type: str, project: project_utils.Project
):
    """
    It takes in a list of annotations, the subject type, and the project, and adds the annotations to
    the database

    :param annotations: a dataframe containing the annotations
    :param subj_type: "clip" or "frame"
    :param project: the project object
    """

    # Get the project-specific name of the database
    db_path = project.db_path

    conn = db_utils.create_connection(db_path)

    # Query id and subject type from the subjects table
    subjects_df = pd.read_sql_query("SELECT id, frame_exp_sp_id FROM subjects", conn)

    # Combine annotation and subject information
    annotations_df = pd.merge(
        annotations,
        subjects_df,
        how="left",
        left_on="subject_ids",
        right_on="id",
        validate="many_to_one",
    )

    # Update agg_annotations_clip table
    if subj_type == "clip":

        # Set the columns in the right order
        species_df = pd.read_sql_query(
            "SELECT id as species_id, label FROM species", conn
        )
        species_df["label"] = species_df["label"].apply(
            lambda x: x.replace(" ", "").replace(")", "").replace("(", "").upper()
        )

        # Combine annotation and subject information
        annotations_df = pd.merge(annotations_df, species_df, how="left", on="label")

        annotations_df = annotations_df[
            ["species_id", "how_many", "first_seen", "subject_ids"]
        ]
        annotations_df["species_id"] = annotations_df["species_id"].apply(
            lambda x: int(x) if not np.isnan(x) else x
        )

        # Test table validity
        db_utils.test_table(
            annotations_df, "agg_annotations_clip", keys=["subject_ids"]
        )

        # Add annotations to the agg_annotations_clip table
        db_utils.add_to_table(
            db_path,
            "agg_annotations_clip",
            [(None,) + tuple(i) for i in annotations_df.values],
            5,
        )

    # Update agg_annotations_frame table
    if subj_type == "frame":

        # Select relevant columns
        annotations_df = annotations_df[["label", "x", "y", "w", "h", "subject_ids"]]

        # Set the columns in the right order
        species_df = pd.read_sql_query(
            "SELECT id as species_id, label FROM species", conn
        )
        species_df["label"] = species_df["label"].apply(
            lambda x: x[:-1] if x == "Blue mussels" else x
        )

        # Combine annotation and subject information
        annotations_df = pd.merge(annotations_df, species_df, how="left", on="label")

        annotations_df = annotations_df[
            ["species_id", "x", "y", "w", "h", "subject_ids"]
        ].dropna()

        # Test table validity

        db_utils.test_table(
            annotations_df, "agg_annotations_frame", keys=["species_id"]
        )

        # Add values to agg_annotations_frame
        db_utils.add_to_table(
            db_path,
            "agg_annotations_frame",
            [(None,) + tuple(i) for i in annotations_df.values],
            7,
        )
