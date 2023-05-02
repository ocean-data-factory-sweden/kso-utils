# base imports
import os
import cv2
import logging
import subprocess
import urllib
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.request import pathname2url
from IPython.display import HTML

# util imports
from kso_utils.server_utils import (
    upload_movie_server,
    get_snic_files,
    get_matching_s3_keys,
)
from kso_utils.project_utils import Project
from kso_utils.tutorials_utils import is_url
from kso_utils.db_utils import create_connection

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


# Function to prevent issues with Swedish characters
# Converting the Swedish characters ä and ö to utf-8.
def unswedify(string: str):
    """Convert ä and ö to utf-8"""
    return (
        string.encode("utf-8")
        .replace(b"\xc3\xa4", b"a\xcc\x88")
        .replace(b"\xc3\xb6", b"o\xcc\x88")
        .decode("utf-8")
    )


# Function to prevent issues with Swedish characters
def reswedify(string: str):
    """Convert ä and ö to utf-8"""
    return (
        string.encode("utf-8")
        .replace(b"a\xcc\x88", b"\xc3\xa4")
        .replace(b"o\xcc\x88", b"\xc3\xb6")
        .decode("utf-8")
    )


def get_fps_duration(movie_path: str):
    """
    This function takes the path (or url) of a movie and returns its fps and duration information

    :param movie_path: a string containing the path (or url) where the movie of interest can be access from
    :return: Two integers, the fps and duration of the movie
    """
    cap = cv2.VideoCapture(movie_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Roadblock to prevent issues with missing movies
    if int(frame_count) | int(fps) == 0:
        raise ValueError(
            f"{movie_path} doesn't have any frames, check the path/link is correct."
        )
    else:
        duration = frame_count / fps

    return fps, duration


def get_movie_path(f_path: str, db_info_dict: dict, project: Project):
    """
    Function to get the path (or url) of a movie

    :param f_path: string with the original path of a movie
    :param db_info_dict: a dictionary with the initial information of the project
    :param project: the project object
    :return: a string containing the path (or url) where the movie of interest can be access from

    """
    # Get the project-specific server
    server = project.server

    if server == "AWS":
        # Extract the key from the orignal path of the movie
        movie_key = urllib.parse.unquote(f_path).split("/", 3)[3]

        # Generate presigned url
        movie_url = db_info_dict["client"].generate_presigned_url(
            "get_object",
            Params={"Bucket": db_info_dict["bucket"], "Key": movie_key},
            ExpiresIn=86400,
        )
        return movie_url

    elif server == "SNIC":
        logging.error("Getting the path of the movies is still work in progress")
        return f_path

    else:
        return f_path


def get_movie_extensions():
    # Specify the formats of the movies to select
    return tuple(["wmv", "mpg", "mov", "avi", "mp4", "MOV", "MP4"])


def standarise_movie_format(
    movie_path: str,
    movie_filename: str,
    f_path: str,
    db_info_dict: dict,
    project: Project,
    gpu_available: bool = False,
):
    """
    This function reviews the movie metadata. If the movie is not in the correct format, frame rate or codec,
    it is converted using ffmpeg.

    :param movie_path: The local path- or url to the movie file you want to convert
    :type movie_path: str
    :param movie_filename: The filename of the movie file you want to convert
    :type movie_filename: str
    :param f_path: The server or storage path of the original movie you want to convert
    :type f_path: str
    :param db_info_dict: a dictionary with the initial information of the project
    :param project: the project object
    :param gpu_available: Boolean, whether or not a GPU is available
    :type gpu_available: bool
    """

    ##### Check movie format ######
    ext = Path(movie_filename).suffix

    if not ext.lower() == "mp4":
        logging.info(f"Extension of {movie_filename} not supported.")
        # Set conversion to True
        convert_video_T_F = True

    ##### Check frame rate #######
    cap = cv2.VideoCapture(movie_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not float(fps).is_integer():
        logging.info(f"Variable frame rate of {movie_filename} not supported.")
        # Set conversion to True
        convert_video_T_F = True

    ##### Check codec info ########
    h = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = (
        chr(h & 0xFF)
        + chr((h >> 8) & 0xFF)
        + chr((h >> 16) & 0xFF)
        + chr((h >> 24) & 0xFF)
    )

    if not codec == "h264":
        logging.info(
            f"The codecs of {movie_filename} are not supported (only h264 is supported)."
        )
        # Set conversion to True
        convert_video_T_F = True

    ##### Check movie file #######
    ##### (not needed in Spyfish) #####
    # Create a list of the project where movie compression is not needed
    project_no_compress = ["Spyfish_Aotearoa"]

    if project.Project_name in project_no_compress:
        # Set movie compression to false
        compress_video = False

    else:
        # Check movie filesize in relation to its duration
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        duration_mins = duration / 60

        # Check if the movie is accessible locally
        if os.path.exists(movie_path):
            # Store the size of the movie
            size = os.path.getsize(movie_path)

        # Check if the path to the movie is a url
        elif is_url(movie_path):
            # Store the size of the movie
            size = urllib.request.urlopen(movie_path).length

        else:
            logging.error(f"The path to {movie_path} is invalid")

        # Calculate the size:duration ratio
        sizeGB = size / (1024 * 1024 * 1024)
        size_duration = sizeGB / duration_mins

        if size_duration > 0.16:
            # Compress the movie if file size is too large
            logging.info(
                "File size of movie is too large (+5GB per 30 mins of footage)."
            )

            # Specify to compress the video
            compress_video = True
        else:
            # Set movie compression to false
            compress_video = False

    # Start converting/compressing video if movie didn't pass any of the checks
    if convert_video_T_F or compress_video:
        conv_mov_path = convert_video(
            movie_path, movie_filename, gpu_available, compress_video
        )

        # Upload the converted movie to the server
        upload_movie_server(conv_mov_path, f_path, db_info_dict, project)

    else:
        logging.info(f"{movie_filename} format is standard.")


def retrieve_movie_info_from_server(project: Project, db_info_dict: dict):
    """
    This function uses the project information and the database information, and returns a dataframe with the
    movie information

    :param project: the project object
    :param db_info_dict: a dictionary containing the path to the database and the client to the server
    :type db_info_dict: dict
    :return: A dataframe with the following columns (index, movie_id, fpath, spath, exists, filename_ext)

    """

    server = project.server
    bucket_i = project.bucket
    movie_folder = project.movie_folder

    if server == "AWS":
        logging.info("Retrieving movies from AWS server")
        # Retrieve info from the bucket
        server_df = get_matching_s3_keys(
            client=db_info_dict["client"],
            bucket=bucket_i,
            suffix=get_movie_extensions(),
        )
        # Get the fpath(html) from the key
        server_df["spath"] = server_df["Key"].apply(
            lambda x: "http://marine-buv.s3.ap-southeast-2.amazonaws.com/"
            + urllib.parse.quote(x, safe="://").replace("\\", "/")
        )

    elif server == "SNIC":
        if "client" in db_info_dict:
            server_df = get_snic_files(
                client=db_info_dict["client"], folder=movie_folder
            )
        else:
            logging.error("No database connection could be established.")
            return pd.DataFrame(columns=["filename"])

    elif server == "LOCAL":
        if [movie_folder, bucket_i] == ["None", "None"]:
            logging.info(
                "No movies to be linked. If you do not have any movie files, please use Tutorial 4 instead."
            )
            return pd.DataFrame(columns=["filename"])
        else:
            server_files = os.listdir(movie_folder)
            server_df = pd.Series(server_files, name="spath").to_frame()
    elif server == "TEMPLATE":
        # Combine wildlife.ai storage and filenames of the movie examples
        server_df = pd.read_csv(db_info_dict["local_movies_csv"])[["filename"]]

        # Get the fpath(html) from the key
        server_df = server_df.rename(columns={"filename": "fpath"})

        server_df["spath"] = server_df["fpath"].apply(
            lambda x: "https://www.wildlife.ai/wp-content/uploads/2022/06/" + str(x), 1
        )

        # Remove fpath values
        server_df.drop(columns=["fpath"], axis=1, inplace=True)

    else:
        raise ValueError("The server type you selected is not currently supported.")

    # Create connection to db
    conn = create_connection(db_info_dict["db_path"])

    # Query info about the movie of interest
    movies_df = pd.read_sql_query("SELECT * FROM movies", conn)
    movies_df = movies_df.rename(columns={"id": "movie_id"})
    
    # If full path is provided, separate into filename and full path
    server_df["spath_full"] = server_df["spath"]
    server_df["spath"] = server_df["spath"].apply(lambda x: Path(x).name, 1)

    # Find closest matching filename (may differ due to Swedish character encoding)
    movies_df["fpath"] = movies_df["fpath"].apply(lambda x: difflib.get_close_matches(
        x, server_df["spath"].unique())[0], 1)
                                                  
    # Merge the server path to the filepath
    movies_df = movies_df.merge(
        server_df,
        left_on=["fpath"],
        right_on=["spath"],
        how="left",
        indicator=True,
    )

    # Check that movies can be mapped
    movies_df["exists"] = np.where(movies_df["_merge"] == "left_only", False, True)
    
    print(movies_df[movies_df.exists == False])

    # Drop _merge columns to match sql schema
    movies_df = movies_df.drop("_merge", axis=1)

    # Select only those that can be mapped
    available_movies_df = movies_df[movies_df["exists"]].copy()

    # Create a filename with ext column
    available_movies_df["filename_ext"] = available_movies_df["spath"].apply(
        lambda x: x.split("/")[-1], 1
    )

    # Add movie folder for SNIC
    if server == "SNIC":
        available_movies_df["spath"] = available_movies_df["spath_full"]

    logging.info(
        f"{available_movies_df.shape[0]} out of {len(movies_df)} movies are mapped from the server"
    )

    return available_movies_df


def convert_video(
    movie_path: str,
    movie_filename: str,
    gpu_available: bool = False,
    compression: bool = False,
):
    """
    It takes a movie file path and a boolean indicating whether a GPU is available, and returns a new
    movie file path.

    :param movie_path: The local path- or url to the movie file you want to convert
    :type movie_path: str
    :param movie_filename: The filename of the movie file you want to convert
    :type movie_path: str
    :param gpu_available: Boolean, whether or not a GPU is available
    :type gpu_available: bool
    :param compression: Boolean, whether or not movie compression is required
    :type compression: bool
    :return: The path to the converted video file.
    """
    movie_filename = Path(movie_path).name
    conv_filename = "conv_" + movie_filename

    # Check the movie is accessible locally
    if os.path.exists(movie_path):
        # Store the directory and filename of the movie
        movie_fpath = os.path.dirname(movie_path)
        conv_fpath = os.path.join(movie_fpath, conv_filename)

    # Check if the path to the movie is a url
    elif is_url(movie_path):
        # Specify the directory to store the converted movie
        conv_fpath = os.path.join(conv_filename)

    else:
        logging.error("The path to", movie_path, " is invalid")

    if gpu_available and compression:
        subprocess.call(
            [
                "ffmpeg",
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-i",
                str(movie_path),
                "-c:v",
                "h264_nvenc",  # ensures correct codec
                "-crf",
                "22",  # compresses the video
                str(conv_fpath),
            ]
        )

    elif gpu_available and not compression:
        subprocess.call(
            [
                "ffmpeg",
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-i",
                str(movie_path),
                "-c:v",
                "h264_nvenc",  # ensures correct codec
                str(conv_fpath),
            ]
        )

    elif not gpu_available and compression:
        subprocess.call(
            [
                "ffmpeg",
                "-i",
                str(movie_path),
                "-c:v",
                "h264",  # ensures correct codec
                "-crf",
                "22",  # compresses the video
                str(conv_fpath),
            ]
        )

    elif not gpu_available and not compression:
        subprocess.call(
            [
                "ffmpeg",
                "-i",
                str(movie_path),
                "-c:v",
                "h264",  # ensures correct codec
                str(conv_fpath),
            ]
        )
    else:
        raise ValueError(f"{movie_path} not modified")

    # Ensure open permissions on file (for now)
    os.chmod(conv_fpath, 0o777)
    logging.info("Movie file successfully converted and stored locally.")
    return conv_fpath


# Function to preview underwater movies
def preview_movie(
    project: Project,
    db_info_dict: dict,
    available_movies_df: pd.DataFrame,
    movie_i: str,
):
    """
    It takes a movie filename and returns a HTML object that can be displayed in the notebook

    :param project: the project object
    :param db_info_dict: a dictionary containing the database information
    :param available_movies_df: a dataframe with all the movies in the database
    :param movie_i: the filename of the movie you want to preview
    :return: A tuple of two elements:
        1. HTML object
        2. Movie path
    """

    # Select the movie of interest
    movie_selected = available_movies_df[
        available_movies_df["filename"] == movie_i
    ].reset_index(drop=True)
    movie_selected_view = movie_selected.T
    movie_selected_view.columns = ["Movie summary"]

    # Make sure only one movie is selected
    if len(movie_selected.index) > 1:
        logging.info(
            "There are several movies with the same filename. This should be fixed!"
        )
        return None

    else:
        # Generate temporary path to the movie select
        if project.server == "SNIC":
            movie_path = get_movie_path(
                project=project,
                db_info_dict=db_info_dict,
                f_path=movie_selected["spath"].values[0],
            )
            url = (
                "https://portal.c3se.chalmers.se/pun/sys/dashboard/files/fs/"
                + pathname2url(movie_path)
            )
        else:
            url = get_movie_path(
                f_path=movie_selected["fpath"].values[0],
                db_info_dict=db_info_dict,
                project=project,
            )
            movie_path = url
        html_code = f"""<html>
                <div style="display: flex; justify-content: space-around; align-items: center">
                <div>
                  <video width=500 controls>
                  <source src={url}>
                  </video>
                </div>
                <div>{movie_selected_view.to_html()}</div>
                </div>
                </html>"""
        return HTML(html_code), movie_path
