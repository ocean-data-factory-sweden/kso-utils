# base imports
import os
import sqlite3
import logging
import pandas as pd
import numpy as np


# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def get_spyfish_col_names(table_name: str):
    """Return a dictionary with the project-specific column names of a csv of interest
    This function helps matching the schema format without modifying the column names of the original csv.

    :param table_name: a string of the name of the schema table of interest
    :return: a dictionary with the names of the columns
    """

    if table_name == "sites":
        # Save the column names of interest in a dict
        col_names_sites = {
            "siteName": "SiteID",
            "decimalLatitude": "Latitude",
            "decimalLongitude": "Longitude",
            "geodeticDatum": "geodeticDatum",
            "countryCode": "countryCode",
        }

        return col_names_sites

    if table_name == "movies":
        # Save the column names of interest in a dict
        col_names_movies = {
            "filename": "filename",
            "created_on": "EventDate",
            "fps": "fps",
            "duration": "duration",
            "sampling_start": "SamplingStart",
            "sampling_end": "SamplingEnd",
            "author": "RecordedBy",
            "SiteID": "SiteID",
            "fpath": "LinkToVideoFile",
        }
        return col_names_movies

    else:
        raise ValueError("The table for Spyfish doesn't match the schema tables")


def process_spyfish_sites(sites_df: pd.DataFrame):
    """
    > This function takes a dataframe of sites and renames the columns to match the schema

    :param sites_df: the dataframe of sites
    :return: A dataframe with the columns renamed.
    """

    # Drop sintemae to avoid duplicated columns
    sites_df = sites_df.drop(columns=["SiteName"])

    # Rename relevant fields
    sites_df = sites_df.rename(
        columns={
            "schema_site_id": "site_id",  # site id for the db
            "SiteID": "siteName",  # site id used for zoo subjects
            "Latitude": "decimalLatitude",
            "Longitude": "decimalLongitude",
        }
    )

    return sites_df


def process_spyfish_movies(movies_df: pd.DataFrame):
    """
    It takes a dataframe of movies and renames the columns to match the columns in the subject metadata
    from Zoo

    :param movies_df: the dataframe containing the movies metadata
    :return: A dataframe with the columns renamed and the file extension removed from the filename.
    """

    # Rename relevant fields
    movies_df = movies_df.rename(
        columns={
            "LinkToVideoFile": "fpath",
            "EventDate": "created_on",
            "SamplingStart": "sampling_start",
            "SamplingEnd": "sampling_end",
            "RecordedBy": "author",
            "SiteID": "siteName",
        }
    )

    # Remove extension from the filename to match the subject metadata from Zoo
    movies_df["filename"] = movies_df["filename"].str.split(".", 1).str[0]

    return movies_df


def process_spyfish_subjects(subjects: pd.DataFrame, db_connection: sqlite3.Connection):
    """
    It takes a dataframe of subjects and a path to the database, and returns a dataframe of subjects
    with the following columns:

    - filename, clip_start_time,clip_end_time,frame_number,subject_type,ScientificName,frame_exp_sp_id,movie_id

    The function does this by:

    - Merging "#Subject_type" and "Subject_type" columns to "subject_type"
    - Renaming columns to match the db format
    - Calculating the clip_end_time
    - Matching 'ScientificName' to species id and save as column "frame_exp_sp_id"
    - Matching site code to name from movies sql and get movie_id to save it as "movie_id"

    :param subjects: the dataframe of subjects to be processed
    :param db_connection: SQL connection object
    :return: A dataframe with the columns:
        - filename, clip_start_time,clip_end_time,frame_number,subject_type,ScientificName,frame_exp_sp_id,movie_id
    """

    if "#Subject_type" in subjects.columns:
        # Merge "#Subject_type" and "Subject_type" columns to "subject_type"
        subjects["#Subject_type"] = subjects["#Subject_type"].fillna(
            subjects["subject_type"]
        )

    if "subject_type" in subjects.columns:
        subjects["subject_type"] = subjects["subject_type"].fillna(
            subjects["#Subject_type"]
        )

    if "Subject_type" in subjects.columns:
        subjects = subjects.rename(columns={"Subject_type": "subject_type"})

    # Create columns to match schema if they don't exist
    subjects["upl_seconds"] = subjects.get("upl_seconds", np.nan)
    subjects["#VideoFilename"] = subjects.get("#VideoFilename", np.nan)
    subjects["#frame_number"] = subjects.get("#frame_number", np.nan)
    subjects["#clip_length"] = subjects.get("#clip_length", np.nan)
    subjects["movie_id"] = subjects.get("movie_id", np.nan)

    # Rename columns to match the db format
    subjects = subjects.rename(
        columns={
            "#VideoFilename": "filename",
            "upl_seconds": "clip_start_time",
            "#frame_number": "frame_number",
        }
    )

    # Calculate the clip_end_time
    subjects["clip_end_time"] = subjects["clip_start_time"] + subjects["#clip_length"]

    from kso_utils.db_utils import get_df_from_db_table

    ##### Match 'ScientificName' to species id and save as column "frame_exp_sp_id"
    if "frame_exp_sp_id" in subjects.columns:
        # Query id and sci. names from the species table
        species_df = get_df_from_db_table(db_connection, "species")[
            ["id", "scientificName"]
        ]

        # Rename columns to match subject df
        species_df = species_df.rename(
            columns={"id": "frame_exp_sp_id", "scientificName": "ScientificName"}
        )

        # Reference the expected species on the uploaded subjects
        subjects = pd.merge(
            subjects.drop(columns=["frame_exp_sp_id"]),
            species_df,
            how="left",
            on="ScientificName",
        )

    else:
        subjects["frame_exp_sp_id"] = np.nan

    ##### Match site code to name from movies sql and get movie_id to save it as "movie_id"
    # Query id and filenames from the movies table
    movies_df = get_df_from_db_table(db_connection, "movies")[["id", "filename"]]

    # Rename columns to match subject df
    movies_df = movies_df.rename(columns={"id": "movie_id"})

    # Drop movie_ids from subjects to avoid issues
    subjects = subjects.drop(columns="movie_id")

    # Reference the movienames with the id movies table
    subjects = pd.merge(subjects, movies_df, how="left", on="filename")

    return subjects


def process_clips_spyfish(annotations, row_class_id, rows_list: list):
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


def spyfish_subject_metadata(df: pd.DataFrame, csv_paths: dict):
    """
    It takes a dataframe of subject metadata and returns a dataframe of subject metadata that is ready
    to be uploaded to Zooniverse

    :param df: the dataframe of all the detections
    :param csv_paths: paths to the csv from the project object
    :return: A dataframe with the columns of interest for uploading to Zooniverse.
    """

    # Get extra movie information
    movies_df = pd.read_csv(csv_paths["local_movies_csv"])

    df = df.merge(movies_df.drop(columns=["filename"]), how="left", on="movie_id")

    # Get extra survey information
    surveys_df = pd.read_csv(csv_paths["local_surveys_csv"])

    df = df.merge(surveys_df, how="left", on="SurveyID")

    # Get extra site information
    sites_df = pd.read_csv(csv_paths["local_sites_csv"])

    df = df.merge(
        sites_df.drop(columns=["LinkToMarineReserve"]), how="left", on="SiteID"
    )

    # Convert datetime to string to avoid JSON seriazible issues
    df["EventDate"] = df["EventDate"].astype(str)

    df = df.rename(
        columns={
            "LinkToMarineReserve": "!LinkToMarineReserve",
            "UID": "#UID",
            "scientificName": "ScientificName",
            "EventDate": "#EventDate",
            "first_seen_movie": "#TimeOfMaxSeconds",
            "frame_number": "#frame_number",
            "filename": "#VideoFilename",
            "SiteID": "#SiteID",
            "SiteCode": "#SiteCode",
            "clip_start_time": "upl_seconds",
        }
    )

    # Select only columns of interest
    upload_to_zoo = df[
        [
            "frame_path",
            "Year",
            "ScientificName",
            "Depth",
            "!LinkToMarineReserve",
            "#EventDate",
            "#TimeOfMaxSeconds",
            "#frame_number",
            "#VideoFilename",
            "#SiteID",
            "#SiteCode",
            "species_id",
        ]
    ].reset_index(drop=True)

    return upload_to_zoo


def add_spyfish_survey_info(movies_df: pd.DataFrame, csv_paths: dict):
    """
    It takes a dataframe of movies and returns it with the survey_specific info

    :param df: the dataframe of all the detections
    :param csv_paths: paths to the csv from the project object
    :return: A dataframe with the columns of interest for uploading to Zooniverse.
    """
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

    return movies_df
