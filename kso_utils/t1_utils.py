# base imports
import os
import json
import subprocess
import pandas as pd
import numpy as np
import time
import logging
from tqdm import tqdm
from pathlib import Path
from csv_diff import compare, load_csv

# widget imports
from IPython.display import display
from ipywidgets import Layout, HBox
import ipywidgets as widgets
from folium.plugins import MiniMap
from ipyfilechooser import FileChooser
import asyncio
import ipysheet
import folium

# util imports
from kso_utils.db_utils import (
    create_connection,
    test_table,
    add_to_table,
    get_column_names_db,
    init_db,
)
from kso_utils.movie_utils import (
    retrieve_movie_info_from_server,
    get_movie_path,
    get_fps_duration,
    standarise_movie_format,
    get_movie_extensions,
)
from kso_utils.server_utils import (
    connect_to_server,
    update_csv_server,
    get_matching_s3_keys,
    upload_file_to_s3,
    download_object_from_s3,
    delete_file_from_s3,
    download_csv_aws,
    download_gdrive,
)
import kso_utils.project_utils as project_utils
from kso_utils.tutorials_utils import wait_for_change

# project-specific imports
from kso_utils.koster_utils import process_koster_movies_csv
from kso_utils.spyfish_utils import (
    process_spyfish_movies,
    process_spyfish_sites,
    check_spyfish_movies,
    get_spyfish_choices,
    get_spyfish_col_names,
)
from kso_utils.sgu_utils import process_sgu_photos_csv


# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

####################################################
############ CSV/iPysheet FUNCTIONS ################
####################################################


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


def update_csv(
    db_info_dict: dict,
    project: project_utils.Project,
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
        layout=Layout(width="25%"),
        style={"description_width": "initial"},
        button_style="danger",
    )

    # Create button to deny changes
    deny_button = widgets.Button(
        description="No, I will go back and fix them",
        layout=Layout(width="45%"),
        style={"description_width": "initial"},
        button_style="danger",
    )

    # Save changes in survey csv locally and in the server
    async def f(sheet_df, df, local_csv, serv_csv):
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
                db_info_dict=db_info_dict, project=project, local_csv=local_csv
            )

            # Log changes locally
            log_meta_changes(
                project,
                db_info_dict=db_info_dict,
                meta_key=local_csv,
                new_sheet_df=sheet_df,
            )

            # Save the updated df locally
            df_orig.to_csv(db_info_dict[local_csv], index=False)
            logging.info("The local csv file has been updated")

            if project.server == "AWS":
                # Save the updated df in the server
                update_csv_server(
                    project, db_info_dict, orig_csv=serv_csv, updated_csv=local_csv
                )

        else:
            logging.info("Run this cell again when the changes are correct!")

    print("")
    print("Are the changes above correct?")
    display(HBox([confirm_button, deny_button]))  # <----Display both buttons in an HBox
    asyncio.create_task(f(sheet_df, df, local_csv, serv_csv))


def log_meta_changes(
    project: project_utils.Project,
    db_info_dict: dict,
    meta_key: str,
    new_sheet_df: pd.DataFrame,
):
    """Records changes to csv files in log file (json format)"""

    diff = {
        "timestamp": int(time.time()),
        "change_info": compare(
            {
                int(k): v
                for k, v in pd.read_csv(db_info_dict[meta_key]).to_dict("index").items()
            },
            {int(k): v for k, v in new_sheet_df.to_dict("index").items()},
        ),
    }

    if len(diff) == 0:
        logging.info("No changes were logged")
        return

    else:
        try:
            with open(Path(project.csv_folder, "change_log.json"), "r+") as f:
                try:
                    existing_data = json.load(f)
                except json.decoder.JSONDecodeError:
                    existing_data = []
                existing_data.append(diff)
                f.seek(0)
                json.dump(existing_data, f)
        except FileNotFoundError:
            with open(Path(project.csv_folder, "change_log.json"), "w") as f:
                json.dump([diff], f)
        logging.info(
            f"Changelog updated at: {Path(project.csv_folder, 'change_log.json')}"
        )
        return


####################################################
############### SITES FUNCTIONS ###################
####################################################
def map_site(db_info_dict: dict, project: project_utils.Project):
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


####################################################
############### MOVIES FUNCTIONS ###################
####################################################


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


def check_movies_csv(
    db_info_dict: dict,
    available_movies_df: pd.DataFrame,
    project: project_utils.Project,
    review_method: str,
    gpu_available: bool = False,
):
    """
    > The function `check_movies_csv` loads the csv with movies information and checks if it is empty

    :param db_info_dict: a dictionary with the following keys:
    :param available_movies_df: a dataframe with all the movies in the database
    :param project: The project name
    :param review_method: The method used to review the movies
    :param gpu_available: Boolean, whether or not a GPU is available
    """

    # Load the csv with movies information
    df = pd.read_csv(db_info_dict["local_movies_csv"])

    # Get project-specific column names
    col_names = get_col_names(project, "local_movies_csv")

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
            df_missing = df[df[col_fps].isna() | df[col_duration].isna()].reset_index(
                drop=True
            )

            ##### Select only movies that can be mapped ####
            # Merge the missing fps/duration df with the available movies
            df_missing = df_missing.merge(
                available_movies_df[["filename", "exists", "spath"]],
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

            #             # Add a column with the path (or url) where the movies can be accessed from
            #             df_missing["movie_path"] = pd.Series(
            #                 [
            #                     movie_utils.get_movie_path(i, db_info_dict, project)
            #                     for i in tqdm(df_missing[col_fpath], total=df_missing.shape[0])
            #                 ]
            #             )
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
                    get_fps_duration(i)
                    for i in tqdm(df_missing["movie_path"], total=df_missing.shape[0])
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
                get_movie_path(i, db_info_dict, project)
                for i in tqdm(df[col_fpath], total=df.shape[0])
            ]
        )

        logging.info("Getting the fps and duration of the movies")
        # Read the movies and overwrite the existing fps and duration info
        df[[col_fps, col_duration]] = pd.DataFrame(
            [get_fps_duration(i) for i in tqdm(df["movie_path"], total=df.shape[0])],
            columns=[col_fps, col_duration],
        )

        logging.info("Standardising the format, frame rate and codec of the movies")

        # Convert movies to the right format, frame rate or codec and upload them to the project's server/storage
        [
            standarise_movie_format(i, j, k, db_info_dict, project, gpu_available)
            for i, j, k in tqdm(
                zip(df["movie_path"], df["filename"], df[col_fpath]), total=df.shape[0]
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
    df.to_csv(db_info_dict["local_movies_csv"], index=False)
    logging.info("The local movies.csv file has been updated")

    # Save the updated df in the server
    update_csv_server(
        project,
        db_info_dict,
        orig_csv="server_movies_csv",
        updated_csv="local_movies_csv",
    )


def check_movies_from_server(db_info_dict: dict, project: project_utils.Project):
    """
    It takes in a dataframe of movies and a dictionary of database information, and returns two
    dataframes: one of movies missing from the server, and one of movies missing from the csv

    :param db_info_dict: a dictionary with the following keys:
    :param project: the project object
    """
    # Load the csv with movies information
    movies_df = pd.read_csv(db_info_dict["local_movies_csv"])

    # Check if the project is the Spyfish Aotearoa
    if project.Project_name == "Spyfish_Aotearoa":
        # Retrieve movies that are missing info in the movies.csv
        missing_info = check_spyfish_movies(movies_df, db_info_dict)

    # Find out files missing from the Server
    missing_from_server = missing_info[missing_info["_merge"] == "left_only"]

    logging.info(f"There are {len(missing_from_server.index)} movies missing")

    # Find out files missing from the csv
    missing_from_csv = missing_info[missing_info["_merge"] == "right_only"].reset_index(
        drop=True
    )

    logging.info(
        f"There are {len(missing_from_csv.index)} movies missing from movies.csv. Their filenames are:{missing_from_csv.filename.unique()}"
    )

    return missing_from_server, missing_from_csv


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


def update_new_deployments(
    deployment_selected: widgets.Widget, db_info_dict: dict, event_date: widgets.Widget
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
        movies_s3_pd = get_matching_s3_keys(
            db_info_dict["client"],
            db_info_dict["bucket"],
            prefix=deployment_i,
            suffix=get_movie_extensions(),
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
                    download_object_from_s3(
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
            upload_file_to_s3(
                db_info_dict["client"],
                bucket=db_info_dict["bucket"],
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
                delete_file_from_s3(
                    client=db_info_dict["client"],
                    bucket=db_info_dict["bucket"],
                    key=movie_i,
                )


#############
#####Species#####
#################
def check_species_csv(db_info_dict: dict, project: project_utils.Project):
    """
    > The function `check_species_csv` loads the csv with species information and checks if it is empty

    :param db_info_dict: a dictionary with the following keys:
    :param project: The project name
    """
    # Load the csv with movies information
    species_df = pd.read_csv(db_info_dict["local_species_csv"])

    # Retrieve the names of the basic columns in the sql db
    conn = create_connection(db_info_dict["db_path"])
    data = conn.execute("SELECT * FROM species")
    field_names = [i[0] for i in data.description]

    # Select the basic fields for the db check
    df_to_db = species_df[[c for c in species_df.columns if c in field_names]]

    # Roadblock to prevent empty lat/long/datum/countrycode
    test_table(df_to_db, "species", df_to_db.columns)

    logging.info("The species dataframe is complete")


def process_test_csv(
    db_info_dict: dict, project: project_utils.Project, local_csv: str
):
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
        field_names, csv_i, df = process_sites_df(db_info_dict, df, project)

    if "movies" in local_csv:
        field_names, csv_i, df = process_movies_df(db_info_dict, df, project)

    if "species" in local_csv:
        field_names, csv_i, df = process_species_df(db_info_dict, df, project)

    if "photos" in local_csv:
        field_names, csv_i, df = process_photos_df(db_info_dict, df, project)

    # Add the names of the basic columns in the sql db
    field_names = field_names + get_column_names_db(db_info_dict, csv_i)
    field_names.remove("id")

    # Select relevant fields
    df.rename(columns={"Author": "author"}, inplace=True)
    df = df[[c for c in field_names if c in df.columns]]

    # Roadblock to prevent empty rows in id_columns
    test_table(df, csv_i, [df.columns[0]])

    return csv_i, df


def process_sites_df(
    db_info_dict: dict, df: pd.DataFrame, project: project_utils.Project
):
    """
    > This function processes the sites dataframe and returns a string with the category of interest

    :param db_info_dict: The dictionary containing the database information
    :param df: a pandas dataframe of the information of interest
    :param project: The project object
    :return: a string of the category of interest and the processed dataframe
    """

    # Check if the project is the Spyfish Aotearoa
    if project.Project_name == "Spyfish_Aotearoa":
        # Rename columns to match schema fields
        df = process_spyfish_sites(df)

    # Specify the category of interest
    csv_i = "sites"

    # Specify the id of the df of interest
    field_names = ["site_id"]

    return field_names, csv_i, df


def get_db_init_info(project: project_utils.Project, server_dict: dict) -> dict:
    """
    This function downloads the csv files from the server and returns a dictionary with the paths to the
    csv files

    :param project: the project object
    :param server_dict: a dictionary containing the server information
    :type server_dict: dict
    :return: A dictionary with the paths to the csv files with the initial info to build the db.
    """

    # Define the path to the csv files with initial info to build the db
    db_csv_info = project.csv_folder

    # Get project-specific server info
    server = project.server
    project_name = project.Project_name

    # Create the folder to store the csv files if not exist
    if not os.path.exists(db_csv_info):
        Path(db_csv_info).mkdir(parents=True, exist_ok=True)
        # Recursively add permissions to folders created
        [os.chmod(root, 0o777) for root, dirs, files in os.walk(db_csv_info)]

    if server == "AWS":
        # Download csv files from AWS
        db_initial_info = download_csv_aws(project_name, server_dict, db_csv_info)

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
        db_initial_info = download_gdrive(gdrive_id, db_csv_info)

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
    db_initial_info["db_path"] = project.db_path
    if "client" in server_dict:
        db_initial_info["client"] = server_dict["client"]

    return db_initial_info


def process_movies_df(
    db_info_dict: dict, df: pd.DataFrame, project: project_utils.Project
):
    """
    > This function processes the movies dataframe and returns a string with the category of interest

    :param db_info_dict: The dictionary containing the database information
    :param df: a pandas dataframe of the information of interest
    :param project: The project object
    :return: a string of the category of interest and the processed dataframe
    """

    # Check if the project is the Spyfish Aotearoa
    if project.Project_name == "Spyfish_Aotearoa":
        df = process_spyfish_movies(df)

    # Check if the project is the KSO
    if project.Project_name == "Koster_Seafloor_Obs":
        df = process_koster_movies_csv(df)

    # Connect to database
    conn = create_connection(db_info_dict["db_path"])

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


def process_photos_df(
    db_info_dict: dict, df: pd.DataFrame, project: project_utils.Project
):
    """
    > This function processes the photos dataframe and returns a string with the category of interest

    :param db_info_dict: The dictionary containing the database information
    :param df: a pandas dataframe of the information of interest
    :param project: The project object
    :return: a string of the category of interest and the processed dataframe
    """
    # Check if the project is the SGU
    if project.Project_name == "SGU":
        df = process_sgu_photos_csv(db_info_dict)

    # Specify the category of interest
    csv_i = "photos"

    # Specify the id of the df of interest
    field_names = ["ID"]

    return field_names, csv_i, df


def process_species_df(
    db_info_dict: dict, df: pd.DataFrame, project: project_utils.Project
):
    """
    > This function processes the species dataframe and returns a string with the category of interest

    :param db_info_dict: The dictionary containing the database information
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


def populate_db(db_initial_info: dict, project: project_utils.Project, local_csv: str):
    """
    > This function populates a sql table of interest based on the info from the respective csv

    :param db_initial_info: The dictionary containing the initial database information
    :param project: The project object
    :param local_csv: a string of the names of the local csv to populate from
    """

    # Process the csv of interest and tests for compatibility with sql table
    csv_i, df = process_test_csv(
        db_info_dict=db_initial_info, project=project, local_csv=local_csv
    )

    # Add values of the processed csv to the sql table of interest
    add_to_table(
        db_initial_info["db_path"],
        csv_i,
        [tuple(i) for i in df.values],
        len(df.columns),
    )


def get_col_names(project: project_utils.Project, local_csv: str):
    """Return a dictionary with the project-specific column names of a csv of interest
    This function helps matching the schema format without modifying the column names of the original csv.

    :param project: The project object
    :param local_csv: a string of the name of the local csv of interest
    :return: a dictionary with the names of the columns
    """

    # Get project-specific server info
    project_name = project.Project_name

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


def initiate_db(project: project_utils.Project):
    """
    This function takes a project name as input and returns a dictionary with all the information needed
    to connect to the project's database

    :param project: The name of the project. This is used to get the project-specific info from the config file
    :return: A dictionary with the following keys (db_path, project_name, server_i_dict, db_initial_info)

    """

    # Check if template project
    if project.Project_name == "model-registry":
        return {}

    # Get project specific info
    server_i_dict, db_initial_info = get_project_details(project)

    # Check if server and db info
    if len(server_i_dict) == 0 and len(db_initial_info) == 0:
        return {}

    # Initiate the sql db
    init_db(db_initial_info["db_path"])

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
            populate_db(
                db_initial_info=db_initial_info, project=project, local_csv=local_i_csv
            )

    # Combine server/project info in a dictionary
    db_info_dict = {**db_initial_info, **server_i_dict}

    return db_info_dict


def get_project_details(project: project_utils.Project):
    """
    > This function connects to the server (or folder) hosting the csv files, and gets the initial info
    from the database

    :param project: the project object
    """

    # Connect to the server (or folder) hosting the csv files
    server_i_dict = connect_to_server(project)

    # Get the initial info
    db_initial_info = get_db_init_info(project, server_i_dict)

    return server_i_dict, db_initial_info


def choose_footage(
    project: project_utils.Project, start_path: str = ".", folder_type: str = ""
):
    if project.server == "AWS":
        db_info_dict = initiate_db(project)
        available_movies_df = retrieve_movie_info_from_server(
            project=project, db_info_dict=db_info_dict
        )
        movie_dict = {
            name: get_movie_path(f_path, db_info_dict, project)
            for name, f_path in available_movies_df[["filename", "fpath"]].values
        }

        movie_widget = widgets.SelectMultiple(
            options=[(name, movie) for name, movie in movie_dict.items()],
            description="Select movie(s):",
            ensure_option=False,
            disabled=False,
            layout=Layout(width="50%"),
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


#

# def upload_movies():

#     # Define widget to upload the files
#     mov_to_upload = widgets.FileUpload(
#         accept='.mpg',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
#         multiple=True  # True to accept multiple files upload else False
#     )

#     # Display the widget?
#     display(mov_to_upload)

#     main_out = widgets.Output()
#     display(main_out)

#     # TODO Copy the movie files to the movies folder

#     # Provide the site, location, date info of the movies
#     upload_info_movies()
#     print("uploaded")

# # Check that videos can be mapped
#     movies_df['exists'] = movies_df['Fpath'].map(os.path.isfile)

# def upload_info_movies():

#     # Select the way to upload the info about the movies
#     widgets.ToggleButton(
#     value=False,
#     description=['I have a csv file with information about the movies',
#                  'I am happy to write here all the information about the movies'],
#     disabled=False,
#     button_style='success', # 'success', 'info', 'warning', 'danger' or ''
#     tooltip='Description',
#     icon='check'
# )

#     # Upload the information using a csv file
#     widgets.FileUpload(
#     accept='',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
#     multiple=False  # True to accept multiple files upload else False
# )
#     # Upload the information

#     # the folder where the movies are

#     # Try to extract location and date from the movies
#     widgets.DatePicker(
#     description='Pick a Date',
#     disabled=False
# )

#     # Run an interactive way to write metadata info about the movies

#     print("Thanks for providing all the required information about the movies")


# # Select multiple movies to include information of
# def go_pro_movies_to_update(df):

#     # Save the filenames of the movies missing
#     filename_missing_csv = df.location_and_filename.unique()

#     # Display the project options
#     movie_to_update = widgets.SelectMultiple(
#         options=filename_missing_csv,
#         rows=15,
#         layout=Layout(width='80%'),
#         description="GO pro movies:",
#         disabled=False,

#     )

#     display(movie_to_update)
#     return movie_to_update

# # Select one movie to include information of
# def full_movie_to_update(df):

#     # Save the filenames of the movies missing
#     filename_missing_csv = df.location_and_filename.unique()

#     # Display the project options
#     movie_to_update = widgets.Dropdown(
#         options=filename_missing_csv,
#         rows=15,
#         layout=Layout(width='80%'),
#         description="Full movie:",
#         disabled=False,

#     )

#     display(movie_to_update)
#     return movie_to_update


# # Select the info to add to the csv
# def info_to_csv(df, movies):

#     # Save the filenames of the movies missing
#     filename_missing_csv = df.location_and_filename.unique()

#     # Display the project options
#     movie_to_update = widgets.SelectMultiple(
#         options=filename_missing_csv,
#         rows=15,
#         layout=Layout(width='80%'),
#         description="Movie:",
#         disabled=False,

#     )

#     display(movie_to_update)
#     return movie_to_update
