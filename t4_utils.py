# t4 utils
# base imports
import os
import sys
import sqlite3
import ffmpeg as ffmpeg_python
import re
import pandas as pd
import numpy as np
import shutil
import logging
import cv2

# widget imports
from tqdm import tqdm
from IPython.display import display, clear_output
import ipywidgets as widgets
from ipyfilechooser import FileChooser
from pathlib import Path
from datetime import date
from panoptes_client import (
    SubjectSet,
    Subject,
)

# util imports
from kso_utils.zooniverse_utils import populate_agg_annotations
import kso_utils.tutorials_utils as t_utils
import kso_utils.db_utils as db_utils
import kso_utils.server_utils as s_utils
import kso_utils.koster_utils as k_utils
import kso_utils.spyfish_utils as spyfish_utils
import kso_utils.project_utils as project_utils
import kso_utils.movie_utils as movie_utils
import kso_utils.t8_utils as t8

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# Specify volume allocated by SNIC
snic_path = "/mimer/NOBACKUP/groups/snic2022-22-1210"


def choose_species(zoo_info_dict: dict):
    """
    This function generates a widget to select the species of interest.
    The options visible in the widgets are all the names that are given in the annotations from Zooniverse.
    :param zoo_info_dict: a dictionary containing the information of the subjects uploaded to Zooniverse 
    :type zoo_info_dict: dict
    """
    species = set()

    all_annotations = zoo_info_dict['classifications']['annotations']
    for index in range(len(all_annotations)):
      annotation =  eval(all_annotations[index])[0]['value']
      for ii in range(len(annotation)): 
        species.add(annotation[ii]['choice'])

    species_list = list(species)

    # Roadblock to check if species list is empty
    if len(species_list) == 0:
        raise ValueError(
            f"Your annotations do not contain any species, please check the annotations before continuing."
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


# Function to choose a folder path
def choose_folder():
    fc = FileChooser(".")
    display(fc)
    return fc


# Function to clean label (no non-alpha characters)
def clean_label(label_string: str):
    label_string = label_string.upper()
    label_string = label_string.replace(" ", "")
    pattern = r"[^A-Za-z0-9]+"
    cleaned_string = re.sub(pattern, "", label_string)
    return cleaned_string


# Function to match species selected to species id
def get_species_ids(project: project_utils.Project, species_list: list):
    """
    # Get ids of species of interest
    """
    db_path = project.db_path
    conn = db_utils.create_connection(db_path)
    if len(species_list) == 1:
        species_ids = pd.read_sql_query(
            f'SELECT id FROM species WHERE label=="{species_list[0]}"', conn
        )["id"].tolist()
    else:
        species_ids = pd.read_sql_query(
            f"SELECT id FROM species WHERE label IN {tuple(species_list)}", conn
        )["id"].tolist()
    return species_ids


def get_species_frames(
    agg_clips_df: pd.DataFrame,
    species_ids: list,
    server_dict: dict,
    conn: sqlite3.Connection,
    project: project_utils.Project,
    n_frames_subject: int,
):
    """
    # Function to identify up to n number of frames per classified clip
    # that contains species of interest after the first time seen

    # Find classified clips that contain the species of interest
    """

    # Retrieve list of subjects
    subjects_df = pd.read_sql_query(
        "SELECT id, clip_start_time, movie_id FROM subjects WHERE subject_type='clip'",
        conn,
    )
    
    agg_clips_df["subject_ids"] = agg_clips_df["subject_ids"].astype(int)
    subjects_df["id"] = subjects_df["id"].astype(int)

    # Combine the aggregated clips and subjects dataframes
    frames_df = pd.merge(
        agg_clips_df, subjects_df, how="left", left_on="subject_ids", right_on="id"
    ).drop(columns=["id"])

    # Identify the second of the original movie when the species first appears
    frames_df["first_seen_movie"] = (
        frames_df["clip_start_time"] + frames_df["first_seen"]
    )

    server = project.server

    movies_df = s_utils.retrieve_movie_info_from_server(project, server_dict)
    movie_folder = project.movie_folder

    # Include movies' filepath and fps to the df
    frames_df = frames_df.merge(movies_df, left_on="movie_id", right_on="id")
    frames_df["fpath"] = frames_df["spath"]

    if len(frames_df[~frames_df.exists]) > 0:
        logging.error(
            f"There are {len(frames_df) - frames_df.exists.sum()} out of {len(frames_df)} frames with a missing movie"
        )

    # Select only frames from movies that can be found
    frames_df = frames_df[frames_df.exists]
    if len(frames_df) == 0:
        logging.error(
            "There are no frames for this species that meet your aggregation criteria."
            "Please adjust your aggregation criteria / species choice and try again."
        )

    ##### Add species_id info ####
    # Retrieve species info
    species_df = pd.read_sql_query(
        "SELECT id, label, scientificName FROM species",
        conn,
    )

    # Retrieve species info
    species_df = species_df.rename(columns={"id": "species_id"})

    # Match format of species name to Zooniverse labels
    species_df["label"] = species_df["label"].apply(clean_label)

    # Combine the aggregated clips and subjects dataframes
    frames_df = pd.merge(frames_df, species_df, how="left", on="label").drop(
        columns=["id"]
    )

    # Identify the ordinal number of the frames expected to be extracted
    if len(frames_df) == 0:
        raise ValueError("No frames. Workflow stopped.")

    frames_df["frame_number"] = frames_df[["first_seen_movie", "fps"]].apply(
        lambda x: [
            int((x["first_seen_movie"] + j) * x["fps"]) for j in range(n_frames_subject)
        ],
        1,
    )

    # Reshape df to have each frame as rows
    lst_col = "frame_number"

    frames_df = pd.DataFrame(
        {
            col: np.repeat(frames_df[col].values, frames_df[lst_col].str.len())
            for col in frames_df.columns.difference([lst_col])
        }
    ).assign(**{lst_col: np.concatenate(frames_df[lst_col].values)})[
        frames_df.columns.tolist()
    ]

    # Drop unnecessary columns
    frames_df.drop(["subject_ids"], inplace=True, axis=1)

    return frames_df


# Function to gather information of frames already uploaded
def check_frames_uploaded(
    frames_df: pd.DataFrame,
    project: project_utils.Project,
    species_ids: list,
    conn: sqlite3.Connection,
):
    if project.server == "SNIC":
        # Get info of frames of the species of interest already uploaded
        if len(species_ids) <= 1:
            uploaded_frames_df = pd.read_sql_query(
                f"SELECT movie_id, frame_number, \
            frame_exp_sp_id FROM subjects WHERE frame_exp_sp_id=='{species_ids[0]}' AND subject_type='frame'",
                conn,
            )

        else:
            uploaded_frames_df = pd.read_sql_query(
                f"SELECT movie_id, frame_number, frame_exp_sp_id FROM subjects WHERE frame_exp_sp_id IN \
            {tuple(species_ids)} AND subject_type='frame'",
                conn,
            )

        # Filter out frames that have already been uploaded
        if (
            len(uploaded_frames_df) > 0
            and not uploaded_frames_df["frame_number"].isnull().any()
        ):
            logging.info(
                "There are some frames already uploaded in Zooniverse for the species selected. \
                  Checking if those are the frames you are trying to upload"
            )
            # Ensure that frame_number is an integer
            uploaded_frames_df["frame_number"] = uploaded_frames_df[
                "frame_number"
            ].astype(int)
            frames_df["frame_number"] = frames_df["frame_number"].astype(int)
            merge_df = (
                pd.merge(
                    frames_df,
                    uploaded_frames_df,
                    left_on=["movie_id", "frame_number"],
                    right_on=["movie_id", "frame_number"],
                    how="left",
                    indicator=True,
                )["_merge"]
                == "both"
            )

            # Exclude frames that have already been uploaded
            # trunk-ignore(flake8/E712)
            frames_df = frames_df[merge_df == False]
            if len(frames_df) == 0:
                logging.error(
                    "All of the frames you have selected are already uploaded."
                )
            else:
                logging.info(
                    "There are",
                    len(frames_df),
                    "frames with the species of interest not uploaded to Zooniverse yet.",
                )

        else:
            logging.info(
                "There are no frames uploaded in Zooniverse for the species selected."
            )

    return frames_df


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


# Function to extract selected frames from videos
def extract_frames(
    project: project_utils.Project,
    df: pd.DataFrame,
    server_dict: dict,
    frames_folder: str,
):
    """
    Extract frames and save them in chosen folder.
    """
    # Extract server info
    project_name = project.Project_name

    # Set the filename of the frames
    df["frame_path"] = (
        frames_folder
        + df["filename"].astype(str)
        + "_frame_"
        + df["frame_number"].astype(str)
        + "_"
        + df["label"].astype(str)
        + ".jpg"
    )

    # Create the folder to store the frames if not exist
    if not os.path.exists(frames_folder):
        Path(frames_folder).mkdir(parents=True, exist_ok=True)
        # Recursively add permissions to folders created
        [os.chmod(root, 0o777) for root, dirs, files in os.walk(frames_folder)]

    for movie in df["fpath"].unique():
        url = movie_utils.get_movie_path(
            project=project, db_info_dict=server_dict, f_path=movie
        )

        if url is None:
            logging.error(f"Movie {movie} couldn't be found in the server.")
        else:
            # Select the frames to download from the movie
            key_movie_df = df[df["fpath"] == movie].reset_index()

            # Read the movie on cv2 and prepare to extract frames
            write_movie_frames(key_movie_df, url)

        logging.info("Frames extracted successfully")

    return df


# Function to the provide drop-down options to select the frames to be uploaded
def get_frames(
    species_names: list,
    db_path: str,
    zoo_info_dict: dict,
    server_dict: dict,
    project: project_utils.Project,
    n_frames_subject=3,
    subsample_up_to=100,
):
    # Roadblock to check if species list is empty
    if len(species_names) == 0:
        raise ValueError(
            f"No species were selected. Please select at least one species before continuing."
        )

    # Transform species names to species ids
    species_ids = get_species_ids(project, species_names)

    conn = db_utils.create_connection(db_path)

    if project.movie_folder is None:
        # Extract frames of interest from a folder with frames
        if project.server == "SNIC":
            df = FileChooser(f"{snic_path}/tmp_dir")
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
        workflows_out = t8.WidgetMaker(zoo_info_dict["workflows"])
        display(workflows_out)

        # Select the agreement threshold to aggregrate the responses
        agg_params = t8.choose_agg_parameters("clip")

        # Select the temp location to store frames before uploading them to Zooniverse
        if project.server == "SNIC":
            df = FileChooser(f"{snic_path}/tmp_dir")
        else:
            df = FileChooser(".")
        df.title = "<b>Choose location to store frames</b>"

        # Callback function
        def extract_files(chooser):
            # Get the aggregated classifications based on the specified agreement threshold
            clips_df = t8.get_classifications(
                workflows_out.checks,
                zoo_info_dict["workflows"],
                "clip",
                zoo_info_dict["classifications"],
                db_path,
                project,
            )

            agg_clips_df, raw_clips_df = t8.aggregrate_classifications(
                clips_df, "clip", project, agg_params=agg_params
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
            frame_df = get_species_frames(
                sp_agg_clips_df,
                species_ids,
                server_dict,
                conn,
                project,
                n_frames_subject,
            )

            # Check the frames haven't been uploaded to Zooniverse
            frame_df = check_frames_uploaded(frame_df, project, species_ids, conn)

            # Extract the frames from the videos and store them in the temp location
            if project.server == "SNIC":
                folder_name = chooser.selected
                frames_folder = os.path.join(
                    folder_name, "_".join(species_names_zoo) + "_frames/"
                )
            else:
                frames_folder = "_".join(species_names_zoo) + "_frames/"
            chooser.df = extract_frames(project, frame_df, server_dict, frames_folder)

        # Register callback function
        df.register_callback(extract_files)
        display(df)

    return df


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


# Function modify the frames
def modify_frames(
    frames_to_upload_df: pd.DataFrame,
    species_i: list,
    modification_details: dict,
    project: project_utils.Project,
):
    server = project.server

    # Specify the folder to host the modified clips
    if server == "SNIC":
        folder_name = f"{snic_path}/tmp_dir/frames/"
        mod_frames_folder = os.path.join(
            folder_name, "modified_" + "_".join(species_i) + "_frames/"
        )
    else:
        mod_frames_folder = "modified_" + "_".join(species_i) + "_frames/"

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
                init_prompt = f"ffmpeg_python.input('{row['frame_path']}')"
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
                    print(full_prompt)
                    eval(full_prompt).run(capture_stdout=True, capture_stderr=True)
                    os.chmod(row["modif_frame_path"], 0o777)
                except ffmpeg_python.Error as e:
                    logging.info("stdout:", e.stdout.decode("utf8"))
                    logging.info("stderr:", e.stderr.decode("utf8"))
                    raise e

        logging.info("Frames modified successfully")

    else:
        # Save the modification details to include as subject metadata
        frames_to_upload_df["modif_frame_path"] = frames_to_upload_df["frame_path"]

    return frames_to_upload_df


# Function to set the metadata of the frames to be uploaded to Zooniverse
def set_zoo_metadata(
    df, species_list: list, project: project_utils.Project, db_info_dict: dict
):
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
        upload_to_zoo = spyfish_utils.spyfish_subject_metadata(df, db_info_dict)
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
    upload_to_zoo: dict,
    species_list: list,
    db_info_dict: dict,
    project: project_utils.Project,
):
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
        surveys_df = pd.read_csv(db_info_dict["local_surveys_csv"])
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
            + date.today().strftime("_%d_%m_%Y")
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
