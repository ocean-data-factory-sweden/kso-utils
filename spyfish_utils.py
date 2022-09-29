# base imports
import os
import boto3
import logging
import pandas as pd
from tqdm import tqdm
import subprocess
from pathlib import Path

# util imports
import kso_utils.server_utils as server_utils
import kso_utils.movie_utils as movie_utils
import kso_utils.db_utils as db_utils

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


def check_spyfish_movies(movies_df: pd.DataFrame, db_info_dict: dict):
    """
    It takes a dataframe of movies and a dictionary with the info of the database and returns a
    dataframe with the movies that are in the database

    :param movies_df: a dataframe with the movies to be checked
    :type movies_df: pd.DataFrame
    :param db_info_dict: a dictionary with the following keys:
    :type db_info_dict: dict
    :return: A dataframe with the movies that are in the database and in the S3 bucket.
    """
    #     ################# Get survey and site id from the movies csv

    #     # Load the csv with with sites and survey choices
    #     choices_df = pd.read_csv(db_info_dict["local_choices_csv"])

    #     # Read surveys csv
    #     surveys_df = pd.read_csv(db_info_dict["local_surveys_csv"],parse_dates=['SurveyStartDate'])

    #     # Add short name of the marine reserve to the survey df
    #     surveys_df = surveys_df.merge(choices_df[["ShortFolder", "MarineReserve"]],
    #                                  righton="MarineReserve",
    #                                  lefton="LinkToMarineReserve",
    #                                  how="left")

    #     # Add survey info to each movie
    #     movies_df = movies_df.merge(surveys_df,
    #                                 on=['SurveyID'],
    #                                 how='left')

    #     # Add a column with the year of the survey
    #     movies_df["survey_year"] = movies_df["SurveyStartDate"].dt.year.values[0]

    #     # Create a column with the deployment folder each movie should be
    #     movies_df["deployment_folder"] = movies_df["ShortFolder"] + "-buv-" + movies_df["survey_year"] + "/"

    # Get a dataframe of all movies from AWS
    movies_s3_pd = server_utils.get_matching_s3_keys(
        db_info_dict["client"],
        db_info_dict["bucket"],
        suffix=movie_utils.get_movie_extensions(),
    )

    # Specify the key of the movies (path in S3 of the object)
    movies_s3_pd["filename"] = movies_s3_pd.Key.str.split("/").str[-1]

    # Create a column with the deployment folder of each movie
    movies_s3_pd["deployment_folder"] = (
        movies_s3_pd.Key.str.split("/").str[:2].str.join("/")
    )

    #     print(movies_s3_pd.head())
    # Missing info for files in the "buv-zooniverse-uploads"
    movies_df = movies_df.merge(
        movies_s3_pd, on=["filename"], how="outer", indicator=True
    )

    # Check that movies can be mapped
    #     movies_df['exists'] = np.where(movies_df["_merge"]=="left_only", False, True)

    # Drop _merge columns to match sql squema
    #     movies_df = movies_df.drop("_merge", axis=1)

    return movies_df


def add_fps_length_spyfish(
    df: pd.DataFrame, miss_par_df: pd.DataFrame, client: boto3.client
):
    """
    It downloads the movie locally, gets the fps and duration, and then deletes the movie

    :param df: the dataframe containing the movies
    :param miss_par_df: a dataframe containing the movies that are missing fps and duration
    :param client: the boto3 client
    :return: The dataframe with the fps and duration added.
    """

    # Loop through each movie missing fps and duration
    for index, row in tqdm(miss_par_df.iterrows(), total=miss_par_df.shape[0]):
        if not os.path.exists(row["filename"]):
            # Download the movie locally
            server_utils.download_object_from_s3(
                client,
                bucket="marine-buv",
                key=row["Key"],
                filename=row["filename"],
            )

        # Set the fps and duration of the movie
        df.at[index, "fps"], df.at[index, "duration"] = movie_utils.get_length(
            row["filename"]
        )

        # Delete the downloaded movie
        os.remove(row["filename"])

    return df


def process_spyfish_sites(sites_df: pd.DataFrame):
    """
    > This function takes a dataframe of sites and renames the columns to match the schema

    :param sites_df: the dataframe of sites
    :return: A dataframe with the columns renamed.
    """

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


# Function to download go pro videos, concatenate them and upload the concatenated videos to aws
def concatenate_videos(df: pd.DataFrame, session: boto3.Session):
    """
    It takes a dataframe with the following columns:

    - bucket
    - prefix
    - filename
    - go_pro_files

    It downloads the go pro videos from the S3 bucket, concatenates them, and uploads the concatenated
    video to the S3 bucket

    :param df: the dataframe with the information about the videos to concatenate
    :type df: pd.DataFrame
    :param session: the boto3 session object
    """

    # Loop through each survey to find out the raw videos recorded with the GoPros
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        # Select the go pro videos from the "i" survey to concatenate
        list1 = row["go_pro_files"].split(";")
        list_go_pro = [row["prefix"] + "/" + s for s in list1]

        # Start text file and list to keep track of the videos to concatenate
        textfile_name = "a_file.txt"
        textfile = open(textfile_name, "w")
        video_list = []

        logging.info("Downloading", len(list_go_pro), "videos")

        # Download each go pro video from the S3 bucket
        for go_pro_i in tqdm(list_go_pro, total=len(list_go_pro)):

            # Specify the temporary output of the go pro file
            go_pro_output = go_pro_i.split("/")[-1]

            # Download the files from the S3 bucket
            if not os.path.exists(go_pro_output):
                server_utils.download_object_from_s3(
                    session,
                    bucket=row["bucket"],
                    key=go_pro_i,
                    filename=go_pro_output,
                )

                # client.download_file(bucket_i, go_pro_i, go_pro_output)

            # Keep track of the videos to concatenate
            textfile.write("file '" + go_pro_output + "'" + "\n")
            video_list.append(go_pro_output)

        textfile.close()

        concat_video = row["filename"]

        if not os.path.exists(concat_video):

            logging.info("Concatenating ", concat_video)

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
                    # "-an",#removes the audio
                    concat_video,
                ]
            )

        logging.info(concat_video, "concatenated successfully")

        # Upload the concatenated video to the S3
        s3_destination = row["prefix"] + "/" + concat_video
        server_utils.upload_file_to_s3(
            session,
            bucket=row["bucket"],
            key=s3_destination,
            filename=concat_video,
        )

        logging.info(concat_video, "successfully uploaded to", s3_destination)

        # Delete the raw videos downloaded from the S3 bucket
        for f in video_list:
            os.remove(f)

        # Delete the text file
        os.remove(textfile_name)

        # Update the fps and length info
        # movie_utils.get_length(concat_video)

        # Delete the concat video
        os.remove(concat_video)

        logging.info("Temporary files and videos removed")


def process_spyfish_subjects(subjects: pd.DataFrame, db_path: str):
    """
    It takes a dataframe of subjects and a path to the database, and returns a dataframe of subjects
    with the following columns:

    - filename, clip_start_time,clip_end_time,frame_number,subject_type,ScientificName,frame_exp_sp_id,movie_id

    The function does this by:

    - Merging "#Subject_type" and "Subject_type" columns to "subject_type"
    - Renaming columns to match the db format
    - Calculating the clip_end_time
    - Creating a connection to the db
    - Matching 'ScientificName' to species id and save as column "frame_exp_sp_id"
    - Matching site code to name from movies sql and get movie_id to save it as "movie_id"

    :param subjects: the dataframe of subjects to be processed
    :param db_path: the path to the database you want to upload to
    :return: A dataframe with the columns:
        - filename, clip_start_time,clip_end_time,frame_number,subject_type,ScientificName,frame_exp_sp_id,movie_id
    """

    # Merge "#Subject_type" and "Subject_type" columns to "subject_type"
    subjects["#Subject_type"] = subjects["#Subject_type"].fillna(
        subjects["subject_type"]
    )
    subjects["subject_type"] = subjects["Subject_type"].fillna(
        subjects["#Subject_type"]
    )

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

    # Create connection to db
    conn = db_utils.create_connection(db_path)

    ##### Match 'ScientificName' to species id and save as column "frame_exp_sp_id"
    # Query id and sci. names from the species table
    species_df = pd.read_sql_query("SELECT id, scientificName FROM species", conn)

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

    ##### Match site code to name from movies sql and get movie_id to save it as "movie_id"
    # Query id and filenames from the movies table
    movies_df = pd.read_sql_query("SELECT id, filename FROM movies", conn)

    # Rename columns to match subject df
    movies_df = movies_df.rename(columns={"id": "movie_id"})

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
    :return: A list of dictionaries, each dictionary containing the classification id, the label, the
    first seen time and the number of individuals.
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


def get_spyfish_choices(server_dict: dict, db_initial_info: dict, db_csv_info: str):
    """
    > This function downloads the csv with the sites and survey choices from the server and saves it
    locally

    :param server_dict: a dictionary containing the server information
    :param db_initial_info: a dictionary with the following keys:
    :param db_csv_info: the local path to the folder where the csv files will be downloaded
    :return: The db_initial_info dictionary with the server and local paths of the choices csv
    """
    # Get the server path of the csv with sites and survey choices
    server_choices_csv = server_utils.get_matching_s3_keys(
        server_dict["client"],
        db_initial_info["bucket"],
        prefix=db_initial_info["key"] + "/" + "choices",
    )["Key"][0]

    # Specify the local path for the csv
    local_choices_csv = str(Path(db_csv_info, Path(server_choices_csv).name))

    # Download the csv
    server_utils.download_object_from_s3(
        server_dict["client"],
        bucket=db_initial_info["bucket"],
        key=server_choices_csv,
        filename=local_choices_csv,
    )

    db_initial_info["server_choices_csv"] = server_choices_csv
    db_initial_info["local_choices_csv"] = Path(local_choices_csv)

    return db_initial_info


def spyfish_subject_metadata(df: pd.DataFrame, db_info_dict: dict):
    """
    It takes a dataframe of subject metadata and returns a dataframe of subject metadata that is ready
    to be uploaded to Zooniverse

    :param df: the dataframe of all the detections
    :param db_info_dict: a dictionary containing the following keys:
    :return: A dataframe with the columns of interest for uploading to Zooniverse.
    """

    # Get extra movie information
    movies_df = pd.read_csv(db_info_dict["local_movies_csv"])

    df = df.merge(movies_df.drop(columns=["filename"]), how="left", on="movie_id")

    # Get extra survey information
    surveys_df = pd.read_csv(db_info_dict["local_surveys_csv"])

    df = df.merge(surveys_df, how="left", on="SurveyID")

    # Get extra site information
    sites_df = pd.read_csv(db_info_dict["local_sites_csv"])

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
