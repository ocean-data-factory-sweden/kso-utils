# base imports
import pandas as pd
import datetime
import os
import subprocess
import logging

# widget imports
from IPython.display import display
from ipywidgets import Layout
import ipywidgets as widgets

# util imports
import kso_utils.movie_utils as movie_utils
import kso_utils.spyfish_utils as spyfish_utils
import kso_utils.server_utils as server_utils
import kso_utils.project_utils as project_utils

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def check_movies_from_server(db_info_dict: dict, project: project_utils.Project):
    """
    It takes in a dataframe with movies information and a dictionary with the database information, and
    returns two dataframes: one with the movies that are missing from the server, and one with the
    movies that are missing from the csv

    :param db_info_dict: a dictionary with the following keys:
    :param project: the project object
    """
    # Load the csv with movies information
    movies_df = pd.read_csv(db_info_dict["local_movies_csv"])

    # Check if the project is the Spyfish Aotearoa
    if project.Project_name == "Spyfish_Aotearoa":
        # Retrieve movies that are missing info in the movies.csv
        missing_info = spyfish_utils.check_spyfish_movies(movies_df, db_info_dict)

    # Find out files missing from the Server
    missing_from_server = missing_info[missing_info["_merge"] == "left_only"]

    logging.info("There are", len(missing_from_server.index), "movies missing")

    # Find out files missing from the csv
    missing_from_csv = missing_info[missing_info["_merge"] == "right_only"].reset_index(
        drop=True
    )

    logging.info(
        "There are", len(missing_from_csv.index), "movies missing from movies.csv"
    )

    #     print("There are", len(missing_from_csv.index), "movies missing from movies.csv. Their filenames are:")
    #     print(*missing_from_csv.filename.unique(), sep = "\n")

    return missing_from_server, missing_from_csv


def select_deployment(missing_from_csv: pd.DataFrame):
    """
    > This function takes a dataframe of missing files and returns a widget that allows the user to
    select the deployment of interest

    :param missing_from_csv: a dataframe of the files that are in the data folder but not in the csv
    file
    :return: A widget object
    """
    if missing_from_csv.shape[0] > 0:
        # Widget to select the deployment of interest
        deployment_widget = widgets.SelectMultiple(
            options=missing_from_csv.deployment_folder.unique(),
            description="New deployment:",
            disabled=False,
            rows=10,
            layout=Layout(width="80%"),
            style={"description_width": "initial"},
        )
        display(deployment_widget)

        return deployment_widget


def select_eventdate():
    """
    > This function creates a date picker widget that allows the user to select a date.
    The function is called `select_eventdate()` and it returns a date picker widget.
    :return: The date widget
    """
    # Select the date
    date_widget = widgets.DatePicker(
        description="Date of deployment:",
        value=datetime.date.today(),
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(date_widget)

    return date_widget


def update_new_deployments(
    deployment_selected, db_info_dict: dict, event_date: widgets.Widget
):
    """
    > The function `update_new_deployments` takes a list of deployments, a dictionary with the database
    information, and an event date and concatenates the movies inside each deployment

    :param deployment_selected: the deployment you want to concatenate
    :param db_info_dict: a dictionary with the following keys:
    :type db_info_dict: dict
    :param event_date: the date of the event you want to concatenate
    """
    for deployment_i in deployment_selected.value:
        logging.info(
            f"Starting to concatenate {deployment_i} out of {len(deployment_selected.value)} deployments selected"
        )

        # Get a dataframe of movies from the deployment
        movies_s3_pd = server_utils.get_matching_s3_keys(
            db_info_dict["client"],
            db_info_dict["bucket"],
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
            logging.info("The files", movie_files_server, "will be concatenated")

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
                        client=db_info_dict["client"],
                        bucket=db_info_dict["bucket"],
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
                        "-c",
                        "copy",
                        filename,
                    ]
                )

            # Upload the concatenated video to the S3
            server_utils.upload_file_to_s3(
                db_info_dict["client"],
                bucket=db_info_dict["bucket"],
                key=deployment_i + "/" + filename,
                filename=filename,
            )

            logging.info(filename, "successfully uploaded to", deployment_i)

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
                    client=db_info_dict["client"],
                    bucket=db_info_dict["bucket"],
                    key=movie_i,
                )
