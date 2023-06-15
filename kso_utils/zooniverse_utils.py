## Zooniverse utils

# base imports
import io
import re
import os
import getpass
import pandas as pd
import json
import logging
import numpy as np
import gdown
import datetime
import ffmpeg
import sqlite3
import shutil
from tqdm import tqdm
from panoptes_client import Panoptes, panoptes, Subject, SubjectSet
from panoptes_client import Project as zooProject
from ast import literal_eval
from ipyfilechooser import FileChooser
from pathlib import Path

# util imports
from kso_utils.project_utils import Project
import kso_utils.db_utils as db_utils
import kso_utils.tutorials_utils as t_utils
import kso_utils.movie_utils as movie_utils

# Widget imports
from IPython.display import display
import ipywidgets as widgets


# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def zoo_credentials():
    zoo_user = getpass.getpass("Enter your Zooniverse user")
    zoo_pass = getpass.getpass("Enter your Zooniverse password")
    return zoo_user, zoo_pass


class AuthenticationError(Exception):
    pass


def connect_zoo_project(project: Project):
    """
    It takes a project name as input, and returns a Zooniverse project object

    :param project: the KSO project you are working
    :return: A Zooniverse project object.
    """
    # Save your Zooniverse user name and password.
    zoo_user, zoo_pass = zoo_credentials()

    # Get the project-specific zooniverse number
    project_n = project.Zooniverse_number

    # Connect to the Zooniverse project
    zoo_project = auth_session(zoo_user, zoo_pass, project_n)

    logging.info("Connected to Zooniverse")

    return zoo_project


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
        project = zooProject(int(float(project_n)))
        return project
    except Exception as e:
        logging.error(e)


# Function to retrieve information from Zooniverse
def retrieve_zoo_info(
    project: Project,
    zoo_project: zooProject,
    zoo_info: str,
    generate_export: bool = False,
):
    """
    This function retrieves the information of interest from Zooniverse and saves it as a pandas data
    frame

    :param project: the kso project object
    :param zoo_project: the Zooniverse project object
    :param zoo_info: a list of the info you want to retrieve from Zooniverse
    :type zoo_info: str
    :param generate_export: boolean determining whether to generate a new export and wait for it to be ready or to just download the latest export
    :return: A dictionary of dataframes.
    """

    if hasattr(project, "info_df"):
        if project.info_df is not None:
            logging.info(
                "Zooniverse info retrieved from cache, to force retrieval set project.info_df = None"
            )
            return project.info_df

    # Create an empty dictionary to host the dfs of interest
    info_df = {}

    for info_n in zoo_info:
        logging.info(f"Retrieving {info_n} from Zooniverse")

        try:
            # Get the information of interest from Zooniverse
            if generate_export:
                try:
                    export = zoo_project.get_export(
                        info_n, generate=generate_export, wait=True, wait_timeout=1800
                    )
                except panoptes.PanoptesAPIException:
                    logging.error(
                        "Export generation time out, retrieving the last available information..."
                    )
                    export = zoo_project.get_export(info_n, generate=False)
            else:
                export = zoo_project.get_export(info_n, generate=generate_export)

            # Save the info as pandas data frame
            try:
                export_df = pd.read_csv(io.StringIO(export.content.decode("utf-8")))
            except pd.errors.ParserError:
                logging.error(
                    "Export retrieval time out, please try again in 1 minute or so."
                )
                export_df = {}
                return
        except:
            logging.info(
                "No connection with Zooniverse, retrieve template info from google drive."
            )
            if info_n == "classifications":
                url = "https://drive.google.com/file/d/1DvJ2nOrG32MR2D7faAJZXMNbEm_ra3rb/view?usp=sharing"
            if info_n == "subjects":
                url = "https://drive.google.com/file/d/18AWRPx3erL25IHekncgKfI_kXHFYAl8e/view?usp=sharing"
            if info_n == "workflows":
                url = "https://drive.google.com/file/d/1bZ6CSxJLxeoX8xVgMU7ZqL76RZDv-09A/view?usp=sharing"
            export = gdown.download(url, info_n + ".csv", quiet=False, fuzzy=True)
            export_df = pd.read_csv(export)

        if len(export_df) > 0:
            # If KSO deal with duplicated subjects
            if project.Project_name == "Koster_Seafloor_Obs":
                from kso_utils.koster_utils import (
                    clean_duplicated_subjects,
                    combine_annot_from_duplicates,
                )

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
        logging.info(f"{info_n} retrieved successfully")

    return info_df


# Function to extract metadata from subjects
def extract_metadata(subj_df: pd.DataFrame):
    """
    The function extracts metadata from a pandas DataFrame and returns two separate DataFrames, one with
    the metadata flattened and one without the metadata.

    :param subj_df: A pandas DataFrame containing subject data, including metadata information in JSON
    format
    :type subj_df: pd.DataFrame
    :return: The function `extract_metadata` returns two dataframes: `subj_df` and `meta_df`. `subj_df`
    is the original input dataframe with the "metadata" and "index" columns dropped, and with the index
    reset. `meta_df` is a new dataframe that contains the flattened metadata information extracted from
    the "metadata" column of the input dataframe.
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
    subjects: pd.DataFrame, project: Project, conn: sqlite3.Connection
):
    """
    Populate the subjects table with the subject metadata

    :param  project: the project object
    :param subjects: the subjects dataframe
    :param conn: SQL connection object

    """

    from kso_utils.koster_utils import process_koster_subjects
    from kso_utils.spyfish_utils import process_spyfish_subjects

    project_name = project.Project_name
    server = project.server
    movie_folder = project.movie_folder

    # Check if the Zooniverse project is the KSO
    if project_name == "Koster_Seafloor_Obs":
        subjects = process_koster_subjects(subjects, conn)

    else:
        # Extract metadata from uploaded subjects
        subjects_df, subjects_meta = extract_metadata(subjects)

        # Combine metadata info with the subjects df
        subjects = pd.concat([subjects_df, subjects_meta], axis=1)

        # Check if the Zooniverse project is the Spyfish
        if project_name == "Spyfish_Aotearoa":
            subjects = process_spyfish_subjects(subjects, conn)

        # If project is not KSO or Spyfish standardise subject info
        else:
            # Create columns to match schema if they don't exist
            subjects["frame_exp_sp_id"] = subjects.get("frame_exp_sp_id", np.nan)
            subjects["frame_number"] = subjects.get("frame_number", np.nan)
            subjects["subject_type"] = subjects.get("subject_type", np.nan)

            # Select only relevant metadata columns
            subjects = subjects[
                [
                    "subject_id",
                    "project_id",
                    "workflow_id",
                    "subject_set_id",
                    "locations",
                    "movie_id",
                    "frame_number",
                    "frame_exp_sp_id",
                    "upl_seconds",
                    "Subject_type",
                    "subject_type",
                    "#VideoFilename",
                    "#clip_length",
                    "classifications_count",
                    "retired_at",
                    "retirement_reason",
                    "created_at",
                ]
            ]

            # Fix weird bug where Subject_type is used instead of subject_type for the column name for some clips
            if "Subject_type" in subjects.columns:
                subjects["subject_type"] = subjects[
                    ["subject_type", "Subject_type"]
                ].apply(lambda x: x[1] if isinstance(x[1], str) else x[0], 1)
                subjects.drop(columns=["Subject_type"], inplace=True)

            # Rename columns to match the db format
            subjects = subjects.rename(
                columns={
                    "#VideoFilename": "filename",
                    "upl_seconds": "clip_start_time",
                    "#frame_number": "frame_number",
                }
            )

            # Remove clip subjects with no clip_start_time info (from different projects)
            subjects = subjects[
                ~(
                    (subjects["subject_type"] == "clip")
                    & (subjects["clip_start_time"].isna())
                )
            ]

            # Calculate the clip_end_time
            subjects["clip_end_time"] = (
                subjects["clip_start_time"] + subjects["#clip_length"]
            )

    # Set subject_id information as id
    subjects = subjects.rename(columns={"subject_id": "id"})

    # Extract the html location of the subjects
    subjects["https_location"] = subjects["locations"].apply(
        lambda x: literal_eval(x)["0"]
    )

    # Set movie_id column to None if no movies are linked to the subject
    if movie_folder == "None" and server in ["LOCAL", "SNIC"]:
        subjects["movie_id"] = None

    # Fix subjects where clip_start_time is not provided but upl_seconds is
    if "clip_start_time" in subjects.columns and "upl_seconds" in subjects.columns:
        subjects["clip_start_time"] = subjects[
            ["clip_start_time", "upl_seconds"]
        ].apply(lambda x: x[0] if not np.isnan(x[0]) else x[1], 1)

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

    # Add a subject type if it is missing
    subjects["subject_type"] = subjects[["clip_start_time", "subject_type"]].apply(
        lambda x: "frame" if np.isnan(x[0]) else "clip", 1
    )
    # Test table validity
    db_utils.test_table(subjects, "subjects", keys=["id"])

    # Add values to subjects
    db_utils.add_to_table(
        conn=conn,
        table_name="subjects",
        values=[tuple(i) for i in subjects.values],
        num_fields=15,
    )

    ##### Print how many subjects are in the db
    # Query id and subject type from the subjects table
    subjects_df = db_utils.get_df_from_db_table(conn, "subjects")[
        ["id", "subject_type"]
    ]
    frame_subjs = subjects_df[subjects_df["subject_type"] == "frame"].shape[0]
    clip_subjs = subjects_df[subjects_df["subject_type"] == "clip"].shape[0]

    logging.info(
        f"The database has a total of "
        f"{frame_subjs}"
        f" frame subjects and "
        f"{clip_subjs}"
        f" clip subjects"
    )


# Relevant for ML and upload frames tutorials
def populate_agg_annotations(
    annotations: pd.DataFrame,
    subj_type: str,
    project: Project,
    conn: sqlite3.Connection,
):
    """
    It takes in a list of annotations, the subject type, and the project, and adds the annotations to
    the database

    :param project: the project object
    :param conn: SQL connection object
    :param annotations: a dataframe containing the annotations
    :param subj_type: "clip" or "frame"
    """

    # Get the project-specific name of the database
    db_path = project.db_path

    # Query id and subject type from the subjects table
    subjects_df = db_utils.get_df_from_db_table(conn, "subjects")[
        ["id", "frame_exp_sp_id"]
    ]

    # Combine annotation and subject information
    annotations_df = pd.merge(
        annotations,
        subjects_df,
        how="left",
        left_on="subject_ids",
        right_on="id",
        validate="many_to_one",
    )

    # Retrieve species info from db
    species_df = db_utils.get_df_from_db_table(conn, "species")[["species_id", "label"]]
    species_df = species_df.rename(columns={"id": "species_id"})

    # Update agg_annotations_clip table
    if subj_type == "clip":
        # Set the columns in the right order
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


def process_clips_template(annotations, row_class_id, rows_list: list):
    """
    For each annotation, if the task is T0, then for each species annotated, flatten the relevant
    answers and save the species of choice, class and subject id.

    :param annotations: the list of annotations for a given subject
    :param row_class_id: the classification id
    :param rows_list: a list of dictionaries, each dictionary is a row in the output dataframe
    :return: A list of dictionaries, each dictionary containing the classification id, the label, the first seen time and the number of individuals.

    """

    for ann_i in annotations:
        if ann_i["task"] == "T0":
            # Select each species annotated and flatten the relevant answers
            for value_i in ann_i["value"]:
                choice_i = {}
                # If choice = 'nothing here', set follow-up answers to blank
                if value_i["choice"] == "NOTHINGHERE":
                    f_time = ""
                    inds = ""
                # If choice = species, flatten follow-up answers
                else:
                    answers = value_i["answers"]
                    for k in answers.keys():
                        if "EARLIESTPOINT" in k:
                            f_time = answers[k].replace("S", "")
                        if "HOWMANY" in k:
                            inds = answers[k]
                            # Deal with +20 fish options
                            if inds == "2030":
                                inds = "25"
                            if inds == "3040":
                                inds = "35"
                        elif "EARLIESTPOINT" not in k and "HOWMANY" not in k:
                            f_time, inds = None, None

                # Save the species of choice, class and subject id
                choice_i.update(
                    {
                        "classification_id": row_class_id,
                        "label": value_i["choice"],
                        "first_seen": f_time,
                        "how_many": inds,
                    }
                )

                rows_list.append(choice_i)

    return rows_list


def set_zoo_clip_metadata(
    project: Project,
    generated_clipsdf: pd.DataFrame,
    sitesdf: pd.DataFrame,
    moviesdf: pd.DataFrame,
):
    """
    This function updates the dataframe of clips to be uploaded with
    metadata about the site and project

    :param project: the project object
    :param generated_clipsdf: a df with the information of the clips to be uploaded
    :param sitesdf: a df with the information of the sites of the project
    :param moviesdf: a df with the information of the movies of the project
    :return: upload_to_zoo, sitename, created_on
    """
    # Add spyfish-specific info
    if project.Project_name == "Spyfish_Aotearoa":
        # Rename the site columns to match standard cols names
        sitesdf = sitesdf.rename(columns={"schema_site_id": "id", "SiteID": "siteName"})

    # Rename the id column to match generated_clipsdf
    sitesdf = sitesdf.rename(columns={"id": "site_id", "siteName": "#siteName"})

    # Combine site info to the generated_clips df
    if "site_id" in generated_clipsdf.columns:
        upload_to_zoo = generated_clipsdf.merge(sitesdf, on="site_id")
        sitename = upload_to_zoo["#siteName"].unique()[0]
    else:
        raise ValueError("Sites table empty. Perhaps try to rebuild the initial db.")

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
    if project.Project_name == "Spyfish_Aotearoa":
        # Rename columns to match schema names
        sitesdf = sitesdf.rename(
            columns={
                "LinkToMarineReserve": "!LinkToMarineReserve",
            }
        )

        # Select only relevant columns
        sitesdf = sitesdf[["!LinkToMarineReserve", "#siteName", "ProtectionStatus"]]

        # Include site info to the df
        upload_to_zoo = upload_to_zoo.merge(sitesdf, on="#siteName")

    if project.Project_name == "Koster_Seafloor_Obs":
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

    logging.info(f"The metadata for the {upload_to_zoo.shape[0]} subjects is ready.")

    return upload_to_zoo, sitename, created_on


def upload_clips_to_zooniverse(
    project: Project,
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
    :param project: the project object
    """

    # Estimate the number of clips
    n_clips = upload_to_zoo.shape[0]

    # Create a new subject set to host the clips
    subject_set = SubjectSet()
    subject_set_name = "clips_" + sitename + "_" + str(int(n_clips)) + "_" + created_on
    subject_set.links.project = project.Zooniverse_number
    subject_set.display_name = subject_set_name
    subject_set.save()

    logging.info(f"{subject_set_name} subject set created")

    # Save the df as the subject metadata
    subject_metadata = upload_to_zoo.set_index("clip_path").to_dict("index")

    # Upload the clips to Zooniverse (with metadata)
    new_subjects = []

    logging.info("Uploading subjects to Zooniverse")
    for clip_path, metadata in tqdm(
        subject_metadata.items(), total=len(subject_metadata)
    ):
        # Create a subject
        subject = Subject()

        # Add project info
        subject.links.project = project.Zooniverse_number

        # Add location of clip
        subject.add_location(clip_path)

        # Add metadata
        subject.metadata.update(metadata)

        # Save subject info
        subject.save()
        new_subjects.append(subject)

    # Upload all subjects
    subject_set.add(new_subjects)

    logging.info("Subjects uploaded to Zooniverse")


def process_clips(project: Project, df: pd.DataFrame):
    """
    This function takes a dataframe of classifications and returns a dataframe of annotations

    :param df: the dataframe of classifications
    :type df: pd.DataFrame
    :return: A dataframe with the classification_id, label, how_many, first_seen, https_location,
            subject_type, and subject_ids.
    """

    from kso_utils.spyfish_utils import (
        process_clips_spyfish,
    )

    # Create an empty list
    rows_list = []

    # Loop through each classification submitted by the users
    for index, row in df.iterrows():
        # Load annotations as json format
        annotations = json.loads(row["annotations"])

        # Select the information from the species identification task
        if project.Zooniverse_number == 9747:
            from kso_utils.koster_utils import process_clips_koster

            rows_list = process_clips_koster(
                annotations, row["classification_id"], rows_list
            )

        # Check if the Zooniverse project is the Spyfish
        if project.Project_name == "Spyfish_Aotearoa":
            rows_list = process_clips_spyfish(
                annotations, row["classification_id"], rows_list
            )

        # Process clips as the default method
        else:
            rows_list = process_clips_template(
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


def aggregate_classifications(
    project: Project, df: pd.DataFrame, subj_type: str, agg_params: list
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
        raw_class_df = process_frames(df)

        # Aggregate frames based on their labels
        agg_labels_df = aggregate_labels(raw_class_df, agg_users, min_users)

        # Get rid of the "empty" labels if other species are among the volunteer consensus
        agg_labels_df = agg_labels_df[
            ~((agg_labels_df["class_n_agg"] > 1) & (agg_labels_df["label"] == "empty"))
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
            from kso_utils.frame_utils import filter_bboxes

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
        raw_class_df = process_clips(project, df)

        # aggregate clips based on their labels
        agg_class_df = aggregate_labels(raw_class_df, agg_users, min_users)

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


# Function to the provide drop-down options to select the frames to be uploaded
def get_frames(
    project: Project,
    zoo_info_dict: dict,
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
    species_ids = t_utils.get_species_ids(project, species_names)

    conn = db_utils.create_connection(project.db_path)

    if project.movie_folder is None:
        # Extract frames of interest from a folder with frames
        if project.server == "SNIC":
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
        workflows_out = WidgetMaker(zoo_info_dict["workflows"])
        display(workflows_out)

        # Select the agreement threshold to aggregrate the responses
        from kso_utils.widgets import choose_agg_parameters

        agg_params = choose_agg_parameters("clip")

        # Select the temp location to store frames before uploading them to Zooniverse
        if project.server == "SNIC":
            # Specify volume allocated by SNIC
            snic_path = "/mimer/NOBACKUP/groups/snic2021-6-9"
            df = FileChooser(str(Path(snic_path, "tmp_dir")))
        else:
            df = FileChooser(".")
        df.title = "<b>Choose location to store frames</b>"

        # Callback function
        def extract_files(chooser):
            # Get the aggregated classifications based on the specified agreement threshold
            clips_df = get_classifications(
                workflows_out.checks,
                zoo_info_dict["workflows"],
                "clip",
                zoo_info_dict["classifications"],
                project.db_path,
            )

            agg_clips_df, raw_clips_df = aggregate_classifications(
                project, clips_df, "clip", agg_params=agg_params
            )

            # Match format of species name to Zooniverse labels
            species_names_zoo = [
                clean_label(species_name) for species_name in species_names
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
            populate_agg_annotations(sp_agg_clips_df, "clip", project)

            # Get df of frames to be extracted
            frame_df = movie_utils.get_species_frames(
                project=project,
                agg_clips_df=sp_agg_clips_df,
                species_ids=species_ids,
                n_frames_subject=n_frames_subject,
            )

            # Check the frames haven't been uploaded to Zooniverse
            frame_df = t_utils.check_frames_uploaded(project, frame_df, species_ids)

            # Extract the frames from the videos and store them in the temp location
            if project.server == "SNIC":
                folder_name = chooser.selected
                frames_folder = Path(
                    folder_name, "_".join(species_names_zoo) + "_frames/"
                )
            else:
                frames_folder = "_".join(species_names_zoo) + "_frames/"
            chooser.df = movie_utils.extract_frames(
                project=project, df=frame_df, frames_folder=frames_folder
            )

        # Register callback function
        df.register_callback(extract_files)
        display(df)

    return df


# Function to set the metadata of the frames to be uploaded to Zooniverse
def set_zoo_frame_metadata(
    project: Project, df: pd.DataFrame, species_list: list, csv_paths=dict
):
    """
    It takes a dataframe of clips or frames, and adds metadata about the site and project to it

    :param df: the dataframe with the media to upload
    :param project: the project object
    :param species_list: a list of the species that should be on the frames
    :param csv_paths: a dictionary with the paths of the csvs used to initiate the db
    :return: upload_to_zoo, sitename, created_on
    """
    project_name = project.Project_name

    if not isinstance(df, pd.DataFrame):
        df = df.df

    if (
        "modif_frame_path" in df.columns
        and "no_modification" not in df["modif_frame_path"].values
    ):
        df["frame_path"] = df["modif_frame_path"]

    # Set project-specific metadata
    if project.Zooniverse_number == 9747 or 9754:
        conn = db_utils.create_connection(project.db_path)
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
        from kso_utils.spyfish_utils import spyfish_subject_metadata

        upload_to_zoo = spyfish_subject_metadata(df, csv_paths=csv_paths)
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
    project: Project,
    upload_to_zoo: pd.DataFrame,
    species_list: list,
):
    """
    It takes a dataframe of frames, and upload it to Zooniverse

    :param df: the dataframe with the media to upload
    :param project: the project object
    :param species_list: a list of the species that should be on the frames
    :return: upload_to_zoo, sitename, created_on
    """

    # Retireve zooniverse project name and number
    project_name = project.Project_name
    project_number = project.Zooniverse_number

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
        surveys_df = pd.read_csv(project.csv_paths["local_surveys_csv"])
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
            from kso_utils.widgets import choose_workflows

            new_widget = choose_workflows(self.workflows_df)
            for wdgt in new_widget:
                wdgt.description = wdgt.description + f" #{_}"
            new_widgets.extend(new_widget)
        self.bool_widget_holder.children = tuple(new_widgets)

    @property
    def checks(self):
        return {w.description: w.value for w in self.bool_widget_holder.children}


# Function modify the frames
def modify_frames(
    project: Project,
    frames_to_upload_df: pd.DataFrame,
    species_i: list,
    modification_details: dict,
):
    server = project.server

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
        if hasattr(project, "output_path"):
            mod_frames_folder = project.output_path + mod_frames_folder

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
        frames_to_upload_df["frame_modification_details"] = str(modification_details)

        # Create the folder to store the videos if not exist
        if not os.path.exists(mod_frames_folder):
            Path(mod_frames_folder).mkdir(parents=True, exist_ok=True)
            # Recursively add permissions to folders created
            [os.chmod(root, 0o777) for root, dirs, files in os.walk(mod_frames_folder)]

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
                    full_prompt += (
                        f".output('{row['modif_frame_path']}', q=20, pix_fmt='yuv420p')"
                    )
                # Run the modification
                try:
                    logging.info(full_prompt)
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


def format_to_gbif_occurence(
    project: Project,
    csv_paths: dict,
    zoo_info_dict: dict,
    df: pd.DataFrame,
    classified_by: str,
    subject_type: str,
):
    """
    > This function takes a df of biological observations classified by citizen scientists, biologists or ML algorithms and returns a df of species occurrences to publish in GBIF/OBIS.
    :param project: the project object
    :param csv_paths: dictionary with the paths of the csv files used to initiate the db
    :param df: the dataframe containing the aggregated classifications
    :param classified_by: the entity who classified the object of interest, either "citizen_scientists", "biologists" or "ml_algorithms"
    :param subject_type: str,
    :param zoo_info_dict: dictionary with the workflow/subjects/classifications retrieved from Zooniverse project
    :return: a df of species occurrences to publish in GBIF/OBIS.
    """

    # If classifications have been created by citizen scientists
    if classified_by == "citizen_scientists":
        #### Retrieve subject information #####
        # Create connection to db
        conn = db_utils.create_connection(project.db_path)

        # Add annotations to db
        populate_agg_annotations(df, subject_type, project)

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
        if "local_surveys_csv" in csv_paths.keys():
            # Read info about the movies
            movies_csv = pd.read_csv(csv_paths["local_movies_csv"])

            # Select only movie ids and survey ids
            movies_csv = movies_csv[["movie_id", "SurveyID"]]

            # Combine the movie_id and survey information
            movies_df = pd.merge(
                movies_df, movies_csv, how="left", left_on="id", right_on="movie_id"
            ).drop(columns=["movie_id"])

            # Read info about the surveys
            surveys_df = pd.read_csv(
                csv_paths["local_surveys_csv"], parse_dates=["SurveyStartDate"]
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
        from kso_utils.tutorials_utils import get_workflow_labels

        commonName_labels_list = [
            get_workflow_labels(zoo_info_dict["workflows"], x, y)
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
        comb_df = pd.merge(comb_df, vernacularName_labels_df, how="left", on="label")

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
            project.Project_name
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


def get_workflow_ids(workflows_df: pd.DataFrame, workflow_names: list):
    # The function that takes a list of workflow names and returns a list of workflow
    # ids.
    return [
        workflows_df[workflows_df.display_name == wf_name].workflow_id.unique()[0]
        for wf_name in workflow_names
    ]


def get_classifications(
    project: Project,
    conn: sqlite3.Connection,
    workflow_dict: dict,
    workflows_df: pd.DataFrame,
    subj_type: str,
    class_df: pd.DataFrame,
):
    """
    It takes in a dictionary of workflows, a dataframe of workflows, the type of subject (frame or
    clip), a dataframe of classifications, the path to the database, and the project name. It returns a
    dataframe of classifications

    :param project: the project object
    :param conn: SQL connection object
    :param workflow_dict: a dictionary of the workflows you want to retrieve classifications for. The
        keys are the workflow names, and the values are the workflow IDs, workflow versions, and the minimum
        number of classifications per subject
    :type workflow_dict: dict
    :param workflows_df: the dataframe of workflows from the Zooniverse project
    :type workflows_df: pd.DataFrame
    :param subj_type: "frame" or "clip"
    :param class_df: the dataframe of classifications from the database
    :return: A dataframe with the classifications for the specified project and workflow.
    """

    names, workflow_versions = [], []
    for i in range(0, len(workflow_dict), 3):
        names.append(list(workflow_dict.values())[i])
        workflow_versions.append(list(workflow_dict.values())[i + 2])

    workflow_ids = get_workflow_ids(workflows_df, names)

    # Filter classifications of interest
    classes = []
    for id, version in zip(workflow_ids, workflow_versions):
        class_df_id = class_df[
            (class_df.workflow_id == id) & (class_df.workflow_version >= version)
        ].reset_index(drop=True)
        classes.append(class_df_id)
    classes_df = pd.concat(classes)

    # Add information about the subject
    # Query id and subject type from the subjects table
    subjects_df = db_utils.get_df_from_db_table(conn, "subjects")

    if subj_type == "frame":
        # Select only frame subjects
        subjects_df = subjects_df[subjects_df["subject_type"] == "frame"]

        # Select columns relevant for frame subjects
        subjects_df = subjects_df[
            [
                "id",
                "subject_type",
                "https_location",
                "filename",
                "frame_number",
                "movie_id",
            ]
        ]

    else:
        # Select only frame subjects
        subjects_df = subjects_df[subjects_df["subject_type"] == "clip"]

        # Select columns relevant for frame subjects
        subjects_df = subjects_df[
            [
                "id",
                "subject_type",
                "https_location",
                "filename",
                "clip_start_time",
                "movie_id",
            ]
        ]

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
    project: Project,
    conn: sqlite3.Connection,
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

    Finally, we call the `aggregrate_classifications` function from the `t8_utils` module, passing
    in the dataframe returned by `get_classifications`, the subject

    :param conn: SQL connection object
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
        # Query id and subject type from the subjects table
        subjects_df = db_utils.get_df_from_db_table(conn, "subjects")

        if subject_type == "frame":
            # Select only frame subjects
            subjects_df = subjects_df[subjects_df["subject_type"] == "frame"]

            # Select columns relevant for frame subjects
            subjects_df = subjects_df[
                [
                    "id",
                    "subject_type",
                    "https_location",
                    "filename",
                    "frame_number",
                    "movie_id",
                ]
            ]

        else:
            # Select only frame subjects
            subjects_df = subjects_df[subjects_df["subject_type"] == "clip"]

            # Select columns relevant for frame subjects
            subjects_df = subjects_df[
                [
                    "id",
                    "subject_type",
                    "https_location",
                    "filename",
                    "clip_start_time",
                    "movie_id",
                ]
            ]

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

    agg_class_df, raw_class_df = aggregate_classifications(
        project,
        get_classifications(classifications_data, subject_type),
        subject_type,
        agg_params,
    )
    if summary:
        agg_class_df = (
            agg_class_df.groupby("label")["subject_ids"].agg("count").to_frame()
        )
    return agg_class_df, raw_class_df
