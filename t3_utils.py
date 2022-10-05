# -*- coding: utf-8 -*-
# base imports
import os
import shutil
import ffmpeg as ffmpeg_python
import pandas as pd
import numpy as np
import math
import datetime
import subprocess
import logging
import random
import threading
from multiprocessing.pool import ThreadPool as Pool

# widget imports
from tqdm import tqdm
from IPython.display import display, clear_output
from ipywidgets import interactive, Layout
import ipywidgets as widgets
from panoptes_client import (
    SubjectSet,
    Subject,
)

# util imports
import kso_utils.db_utils as db_utils
import kso_utils.project_utils as project_utils

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


############################################################
######## Create some clip examples #########################
############################################################


def check_movie_uploaded(movie_i: str, db_info_dict: dict):
    """
    This function takes in a movie name and a dictionary containing the path to the database and returns
    a boolean value indicating whether the movie has already been uploaded to Zooniverse

    :param movie_i: the name of the movie you want to check
    :type movie_i: str
    :param db_info_dict: a dictionary containing the path to the database and the path to the folder
    containing the videos
    :type db_info_dict: dict
    """

    # Create connection to db
    conn = db_utils.create_connection(db_info_dict["db_path"])

    # Query info about the clip subjects uploaded to Zooniverse
    subjects_df = pd.read_sql_query(
        "SELECT id, subject_type, filename, clip_start_time,"
        "clip_end_time, movie_id FROM subjects WHERE subject_type='clip'",
        conn,
    )

    # Save the video filenames of the clips uploaded to Zooniverse
    videos_uploaded = subjects_df.filename.dropna().unique()

    # Check if selected movie has already been uploaded
    already_uploaded = any(mv in movie_i for mv in videos_uploaded)

    if already_uploaded:
        clips_uploaded = subjects_df[subjects_df["filename"].str.contains(movie_i)]
        logging.info(f"{movie_i} has clips already uploaded.")
    else:
        logging.info(f"{movie_i} has not been uploaded to Zooniverse yet")


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


def select_random_clips(movie_i: str, db_info_dict: dict):
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
    conn = db_utils.create_connection(db_info_dict["db_path"])

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
    clip_length_number = interactive(
        n_random_clips,
        clip_length=select_clip_length(),
        n_clips=widgets.IntSlider(
            value=3,
            min=1,
            max=5,
            step=1,
            description="Number of random clips:",
            disabled=False,
            layout=Layout(width="40%"),
            style={"description_width": "initial"},
        ),
    )

    display(clip_length_number)
    return clip_length_number


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


def create_example_clips(
    movie_i: str,
    movie_path: str,
    db_info_dict: dict,
    project: project_utils.Project,
    clip_selection,
    pool_size=4,
):
    """
    This function takes a movie and extracts clips from it, based on the start time and length of the
    clips

    :param movie_i: str, the name of the movie
    :type movie_i: str
    :param movie_path: the path to the movie
    :type movie_path: str
    :param db_info_dict: a dictionary containing the information of the database
    :type db_info_dict: dict
    :param project: the project object
    :param clip_selection: a dictionary with the following keys:
    :param pool_size: The number of parallel processes to run, defaults to 4 (optional)
    :return: The path of the clips
    """

    # Specify the starting seconds and length of the example clips
    clips_start_time = clip_selection.result["clip_start_time"]
    clip_length = clip_selection.result["random_clip_length"]

    # Get project-specific server info
    server = project.server

    # Specify the temp folder to host the clips
    output_clip_folder = movie_i + "_clips"
    if server == "SNIC":
        clips_folder = os.path.join(
            "/cephyr/NOBACKUP/groups/snic2021-6-9/tmp_dir/", output_clip_folder
        )
    else:
        clips_folder = output_clip_folder

    # Create the folder to store the videos if not exist
    if not os.path.exists(clips_folder):
        os.mkdir(clips_folder)
        os.chmod(clips_folder, 0o777)

    # Specify the number of parallel items
    pool = Pool(pool_size)

    # Create empty list to keep track of new clips
    example_clips = []

    # Create the information for each clip and extract it (printing a progress bar)
    for start_time_i in clips_start_time:
        # Create the filename and path of the clip
        output_clip_name = (
            movie_i + "_clip_" + str(start_time_i) + "_" + str(clip_length) + ".mp4"
        )
        output_clip_path = os.path.join(clips_folder, output_clip_name)

        # Add the path of the clip to the list
        example_clips = example_clips + [output_clip_path]

        # Extract the clips and store them in the folder
        pool.apply_async(
            extract_example_clips,
            (
                output_clip_path,
                start_time_i,
                clip_length,
                movie_path,
            ),
        )

    pool.close()
    pool.join()

    logging.info("Clips extracted successfully")
    return example_clips


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
        mod_prompt = ''

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


def create_modified_clips(
    clips_list: list,
    movie_i: str,
    modification_details: dict,
    project: project_utils.Project,
    gpu_available: bool,
    pool_size: int = 4,
):
    """
    This function takes a list of clips, a movie name, a dictionary of modifications, a project, and a
    GPU availability flag, and returns a list of modified clips

    :param clips_list: a list of the paths to the clips you want to modify
    :param movie_i: the path to the movie you want to extract clips from
    :param modification_details: a dictionary with the modifications to be applied to the clips. The
    keys are the names of the modifications and the values are the parameters of the modifications
    :param project: the project object
    :param gpu_available: True if you have a GPU available, False if you don't
    :param pool_size: the number of parallel processes to run, defaults to 4 (optional)
    :return: The modified clips
    """

    # Get project-specific server info
    server = project.server

    # Specify the folder to host the modified clips

    mod_clip_folder = "modified_" + movie_i + "_clips"

    if server == "SNIC":
        mod_clips_folder = os.path.join(
            "/cephyr/NOBACKUP/groups/snic2021-6-9/tmp_dir", mod_clip_folder
        )
    else:
        mod_clips_folder = mod_clip_folder

    # Remove existing modified clips
    if os.path.exists(mod_clips_folder):
        shutil.rmtree(mod_clips_folder)

    if len(modification_details.values()) > 0:

        # Create the folder to store the videos if not exist
        if not os.path.exists(mod_clips_folder):
            os.mkdir(mod_clips_folder)
            os.chmod(str(mod_clips_folder), 0o777)

        # Specify the number of parallel items
        pool = Pool(pool_size)

        # Create empty list to keep track of new clips
        modified_clips = []
        results = []
        # Create the information for each clip and extract it (printing a progress bar)
        for clip_i in clips_list:
            # Create the filename and path of the modified clip
            output_clip_name = "modified_" + os.path.basename(clip_i)
            output_clip_path = os.path.join(mod_clips_folder, output_clip_name)

            # Add the path of the clip to the list
            modified_clips = modified_clips + [output_clip_path]

            # Modify the clips and store them in the folder
            results.append(
                pool.apply_async(
                    modify_clips,
                    (
                        clip_i,
                        modification_details,
                        output_clip_path,
                        gpu_available,
                    ),
                )
            )

        pool.close()
        pool.join()
        return modified_clips
    else:
        logging.info("No modification selected")


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
    extension = os.path.splitext(example_clip_path)[1]

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


############################################################
######## Create the clips to upload to Zooniverse ##########
############################################################


def select_clip_n_len(movie_i: str, db_info_dict: dict):
    """
    This function allows the user to select the length of the clips to upload to the database

    :param movie_i: the name of the movie you want to upload
    :param db_info_dict: a dictionary containing the path to the database and the name of the database
    :return: The number of clips to upload
    """

    # Create connection to db
    conn = db_utils.create_connection(db_info_dict["db_path"])

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
    clip_length_number = interactive(
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
    modifications, and the values are dictionaries containing the details of the modification. The
    details of the modification are:
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
        mod_prompt = ''
        def_output_prompt = f".output('{str(output_clip_path)}', ss={str(upl_second_i)}, t={str(clip_length)}, crf=20, pix_fmt='yuv420p', vcodec='libx264')"

        # Set up modification
        for transform in modification_details.values():
            if "filter" in transform:
                mod_prompt += transform["filter"]

            else:
                # Unnest the modification detail dict
                df = pd.json_normalize(modification_details, sep="_")
                crf = df.filter(regex="crf$", axis=1).values[0][0]
                output_prompt = f".output('{str(output_clip_path)}', crf={crf}, ss={str(upl_second_i)}, t={str(clip_length)}, preset='veryfast', pix_fmt='yuv420p', vcodec='libx264')"

        # Run the modification
        try:
            if len(mod_prompt) > 0:
                full_prompt += mod_prompt
            if output_prompt:
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


def create_clips(
    available_movies_df: pd.DataFrame,
    movie_i: str,
    movie_path: str,
    db_info_dict: dict,
    clip_selection,
    project: project_utils.Project,
    modification_details: dict,
    gpu_available: bool,
    pool_size: int = 4,
):
    """
    This function takes a movie and extracts clips from it

    :param available_movies_df: the dataframe with the movies that are available for the project
    :param movie_i: the name of the movie you want to extract clips from
    :param movie_path: the path to the movie you want to extract clips from
    :param db_info_dict: a dictionary with the database information
    :param clip_selection: a ClipSelection object
    :param project: the project object
    :param modification_details: a dictionary with the following keys:
    :param gpu_available: True or False, depending on whether you have a GPU available to use
    :param pool_size: the number of threads to use to extract the clips, defaults to 4 (optional)
    :return: A dataframe with the clip_path, clip_filename, clip_length, upl_seconds, and
    clip_modification_details
    """

    # Filter the df for the movie of interest
    movie_i_df = available_movies_df[
        available_movies_df["filename"] == movie_i
    ].reset_index(drop=True)

    # Calculate the max number of clips available
    clip_length = clip_selection.kwargs["clip_length"]
    clip_numbers = clip_selection.result
    start_trim = clip_selection.kwargs["clips_range"][0]
    end_trim = clip_selection.kwargs["clips_range"][1]

    # Calculate all the seconds for the new clips to start
    movie_i_df["seconds"] = [
        list(
            range(
                start_trim,
                int(math.floor(end_trim / clip_length) * clip_length),
                clip_length,
            )
        )
    ]

    # Reshape the dataframe with the seconds for the new clips to start on the rows
    potential_start_df = expand_list(movie_i_df, "seconds", "upl_seconds")

    # Specify the length of the clips
    potential_start_df["clip_length"] = clip_length

    if not clip_numbers == potential_start_df.shape[0]:
        logging.info(
            f"There was an issue estimating the starting seconds for the {clip_numbers} clips"
        )

    # Get project-specific server info
    server = project.server

    # Specify the temp folder to host the clips
    temp_clip_folder = movie_i + "_zooniverseclips"
    if server == "SNIC":
        clips_folder = os.path.join(
            "/cephyr/NOBACKUP/groups/snic2021-6-9/tmp_dir/", temp_clip_folder
        )
    else:
        clips_folder = temp_clip_folder

    # Set the filename of the clips
    potential_start_df["clip_filename"] = (
        movie_i
        + "_clip_"
        + potential_start_df["upl_seconds"].astype(str)
        + "_"
        + str(clip_length)
        + ".mp4"
    )

    # Set the path of the clips
    potential_start_df["clip_path"] = (
        clips_folder + os.sep + potential_start_df["clip_filename"]
    )

    # Create the folder to store the videos if not exist
    if not os.path.exists(clips_folder):
        os.mkdir(clips_folder)
        os.chmod(str(clips_folder), 0o777)
    else:
        shutil.rmtree(clips_folder)
        os.mkdir(clips_folder)
        os.chmod(str(clips_folder), 0o777)

    logging.info("Extracting clips")

    for i in range(0, potential_start_df.shape[0], pool_size):
        logging.info(
            f"Modifying {i} to {i+pool_size} out of {potential_start_df.shape[0]}"
        )

        threadlist = []
        # Read each movie and extract the clips
        for index, row in potential_start_df.iloc[i : i + pool_size].iterrows():
            # Extract the videos and store them in the folder
            t = threading.Thread(
                target=extract_clips,
                args=(
                    movie_path,
                    clip_length,
                    row["upl_seconds"],
                    row["clip_path"],
                    modification_details,
                    gpu_available,
                ),
            )
            threadlist.append(t)
            t.start()

        for tr in threadlist:
            tr.join()

    # Add information on the modification of the clips
    potential_start_df["clip_modification_details"] = str(modification_details)

    return potential_start_df


def set_zoo_metadata(
    df: pd.DataFrame, project: project_utils.Project, db_info_dict: dict
):
    """
    It takes a dataframe with clips, and adds metadata about the site and project to it

    :param df: the dataframe with the clips to upload
    :param project: the project object
    :param db_info_dict: a dictionary with the following keys:
    :return: upload_to_zoo, sitename, created_on
    """

    # Create connection to db
    conn = db_utils.create_connection(db_info_dict["db_path"])

    # Query info about the movie of interest
    sitesdf = pd.read_sql_query("SELECT * FROM sites", conn)

    # Combine site info to the df
    if "site_id" in df.columns:
        upload_to_zoo = df.merge(sitesdf, left_on="site_id", right_on="id")
        sitename = upload_to_zoo["siteName"].unique()[0]
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
            "siteName": "#siteName",
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

        # Read sites csv as pd
        sitesdf = pd.read_csv(db_info_dict["local_sites_csv"])

        # Read movies csv as pd
        moviesdf = pd.read_csv(db_info_dict["local_movies_csv"])

        # Rename columns to match schema names
        sitesdf = sitesdf.rename(
            columns={
                "siteName": "SiteID",
            }
        )

        # Include movie info to the sites df
        sitesdf = sitesdf.merge(moviesdf, on="SiteID")

        # Rename columns to match schema names
        sitesdf = sitesdf.rename(
            columns={
                "LinkToMarineReserve": "!LinkToMarineReserve",
                "SiteID": "#SiteID",
            }
        )

        # Select only relevant columns
        sitesdf = sitesdf[["!LinkToMarineReserve", "#SiteID", "ProtectionStatus"]]

        # Include site info to the df
        upload_to_zoo = upload_to_zoo.merge(
            sitesdf, left_on="#siteName", right_on="#SiteID"
        )

    if project.Project_name == "Koster_Seafloor_Obs":

        # Read sites csv as pd
        sitesdf = pd.read_csv(db_info_dict["local_sites_csv"])

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

    return upload_to_zoo, sitename, created_on


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


def upload_clips_to_zooniverse(
    upload_to_zoo: pd.DataFrame,
    sitename: str,
    created_on: str,
    project: project_utils.Project,
):
    """
    It takes a dataframe of clips and metadata, creates a new subject set, and uploads the clips to
    Zooniverse

    :param upload_to_zoo: the dataframe of clips to upload
    :param sitename: the name of the site you're uploading clips from
    :param created_on: the date the clips were created
    :param project: the project ID of the project you want to upload to
    """

    # Estimate the number of clips
    n_clips = upload_to_zoo.shape[0]

    # Create a new subject set to host the clips
    subject_set = SubjectSet()
    subject_set_name = "clips_" + sitename + "_" + str(int(n_clips)) + "_" + created_on
    subject_set.links.project = project
    subject_set.display_name = subject_set_name

    subject_set.save()

    logging.info(subject_set_name, "subject set created")

    # Save the df as the subject metadata
    subject_metadata = upload_to_zoo.set_index("clip_path").to_dict("index")

    # Upload the clips to Zooniverse (with metadata)
    new_subjects = []

    logging.info("Uploading subjects to Zooniverse")
    for modif_clip_path, metadata in tqdm(
        subject_metadata.items(), total=len(subject_metadata)
    ):
        # Create a subject
        subject = Subject()

        # Add project info
        subject.links.project = project

        # Add location of clip
        subject.add_location(modif_clip_path)

        # Add metadata
        subject.metadata.update(metadata)

        # Save subject info
        subject.save()
        new_subjects.append(subject)

    # Upload all subjects
    subject_set.add(new_subjects)

    logging.info("Subjects uploaded to Zooniverse")
