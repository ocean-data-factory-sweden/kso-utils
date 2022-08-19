# base imports
import os
import cv2
import pandas as pd
from tqdm import tqdm
import difflib
import logging
import subprocess

# util imports
import kso_utils.server_utils as server_utils
import kso_utils.project_utils as project_utils

# Logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Calculate length and fps of a movie
def get_length(video_file: str, movie_folder: str):
    
    files = os.listdir(movie_folder)
    
    if os.path.basename(video_file) in files:
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)     
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        length = frame_count/fps
    else:
        logging.error("Length and fps for", video_file, "were not calculated - probably missing")
        length, fps = None, None
        
    return fps, length


def check_fps_duration(db_info_dict: dict, project: project_utils.Project):
    """
    It checks if the fps and duration of the movies are missing from the movies csv file. If they are,
    it retrieves them from the movies and updates the csv file
    
    :param db_info_dict: a dictionary with the following keys:
    :param project: the project object
    :return: the dataframe with the fps and duration information.
    """
    
    movie_folder = project.movie_folder
    
    # Load the csv with movies information
    df = pd.read_csv(db_info_dict["local_movies_csv"])
    
    # Check if fps or duration is missing from any movie
    if not df[["fps", "duration"]].isna().any().all():
        
        logging.info("Fps and duration information checked")
        
    else:

        # Get project server
        server = project.server
        
        # Select only those movies with the missing parameters
        miss_par_df = df[df["fps"].isna()|df["duration"].isna()]
        
        logging.info("Retrieving fps and duration of:")
        logging.info(*miss_par_df.filename.unique(), sep = "\n")
        
        ##### Estimate the fps/duration of the movies ####
        # Add info from AWS
        if server == "AWS":
            # Extract the key of each movie
            miss_par_df['key_movie'] = miss_par_df['LinkToVideoFile'].apply(lambda x: x.split('/',3)[3])
            
            # Loop through each movie missing the info and retrieve it
            for index, row in tqdm(miss_par_df.iterrows(), total=miss_par_df.shape[0]):
                # generate a temp url for the movie 
                url = db_info_dict['client'].generate_presigned_url('get_object', 
                                                                    Params = {'Bucket': db_info_dict['bucket'], 
                                                                              'Key': row['key_movie']}, 
                                                                    ExpiresIn = 100)
                # Calculate the fps and duration
                cap = cv2.VideoCapture(url)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count/fps

                # Update the fps/duration info in the miss_par_df
                miss_par_df.at[index,'fps'] = fps
                miss_par_df.at[index,'duration'] = duration
                
                cap.release()
                
            # Save the fps/duration info in the df
            df["fps"] = df.fps.fillna(miss_par_df.fps)
            df["duration"] = df.duration.fillna(miss_par_df.duration)
            
        else:   
            # Set the fps and duration of each movie
            movie_files = server_utils.get_snic_files(db_info_dict["client"], movie_folder)["spath"].tolist()
            f_movies = pd.Series([difflib.get_close_matches(i, movie_files)[0] for i in df["filename"]])
            full_paths = movie_folder + '/' + f_movies
            df.loc[df["fps"].isna()|df["duration"].isna(), "fps": "duration"] = pd.DataFrame(full_paths.apply(
                                                                                                        lambda x: get_length(x, movie_folder), 1).tolist(), columns=["fps", "duration"])
            df["SamplingStart"] = 0.0
            df["SamplingEnd"] = df["duration"]
            df.to_csv(db_info_dict["local_movies_csv"], index=False)
            
        logging.info("Fps and duration information updated")
        
    return df

     

def check_sampling_start_end(df: pd.DataFrame, db_info_dict: dict):
    """
    This function checks if the sampling start and end times are missing from any movie. If they are, it
    sets the sampling start to 0 and the sampling end to the duration of the movie.
    
    :param df: the dataframe with the movies information
    :param db_info_dict: a dictionary with the following keys:
    :return: The dataframe with the sampling start and end times.
    """
    # Load the csv with movies information
    movies_csv = pd.read_csv(db_info_dict["local_movies_csv"])
    
    # Check if sampling start or end is missing from any movie
    if not df[["sampling_start", "sampling_end"]].isna().all().any():
        
        logging.info("sampling_start and survey_end information checked")
        
    else:
        
        logging.info("Updating the survey_start or survey_end information of:")
        logging.info(*df[df[["sampling_start", "sampling_end"]].isna()].filename.unique(), sep = "\n")
        
        # Set the start of each movie to 0 if empty
        df.loc[df["sampling_start"].isna(),"sampling_start"] = 0

            
        # Set the end of each movie to the duration of the movie if empty
        df.loc[df["survey_end"].isna(),"sampling_end"] = df["duration"]

        # Update the local movies.csv file with the new sampling start/end info
        df.to_csv(movies_csv, index=False)
        
        logging.info("The survey start and end columns have been updated in movies.csv")

        
    # Prevent ending sampling times longer than actual movies
    if (df["sampling_end"] > df["duration"]).any():
        logging.info("The sampling_end times of the following movies are longer than the actual movies")
        logging.info(*df[df["sampling_end"] > df["duration"]].filename.unique(), sep = "\n")

    return df


def get_movie_extensions():
    # Specify the formats of the movies to select
    movie_formats = tuple(['wmv', 'mpg', 'mov', 'avi', 'mp4', 'MOV', 'MP4'])
    return movie_formats


def check_movie_information(movie_path: str):
    """
    This function reviews the movie metadata. If the movie is not in the correct format, frame rate or codec,
    it is converted using ffmpeg. 
    
    :param movie_path: the path to the movie file
    :type movie_path: str
    """

    # Check movie format
    filename, ext = os.path.splitext(movie_path)
    if not ext.lower() == "mp4":
        logging.info("Extension not supported, conversion started.")
        new_mov_path = convert_video(movie_path)

    # Check frame rate
    cap = cv2.VideoCapture(movie_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not float(fps).is_integer():
        logging.info("Variable frame rate not supported, conversion started.")
        new_mov_path = convert_video(movie_path)

    # Check codec information
    h = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = chr(h&0xff) + chr((h>>8)&0xff) + chr((h>>16)&0xff) + chr((h>>24)&0xff)
    if not codec == "h264":
        logging.info("Non-h264 codecs not supported, conversion started.")
        new_mov_path = convert_video(movie_path)

    # Check movie filesize
    size = os.path.getsize(movie_path)
    sizeGB = size/(1024*1024*1024)
    if sizeGB > 5:
        logging.info("File sizes above 5GB are not currently supported, conversion started.")
        new_mov_path = convert_video(movie_path)
           

def convert_video(movie_path: str, gpu_available: bool):
    """
    It takes a movie file path and a boolean indicating whether a GPU is available, and returns a new
    movie file path.
    
    :param movie_path: The path to the movie file you want to convert
    :type movie_path: str
    :param gpu_available: Boolean, whether or not a GPU is available
    :type gpu_available: bool
    :return: The path to the converted video file.
    """

    movie_fpath = os.path.dirname(movie_path)
    movie_filename = os.path.basename(movie_path)
    conv_filename = "conv_" + movie_filename
    conv_fpath = os.path.join(movie_fpath, conv_filename)

    if gpu_available:
        subprocess.call(["ffmpeg",
                    "-hwaccel", "cuda",
                    "-hwaccel_output_format", "cuda", 
                    "-i", str(movie_path), 
                    "-c:v", "h264_nvenc", # ensures correct codec
                    "-an", #removes the audio
                    "-crf", "22", # compresses the video
                    str(conv_fpath)])
    else:
        subprocess.call(["ffmpeg", 
                    "-i", str(movie_path), 
                    "-c:v", "h264", # ensures correct codec
                    "-an", #removes the audio
                    "-crf", "22", # compresses the video
                    str(conv_fpath)])

    # Ensure open permissions on file (for now)
    os.chmod(conv_fpath, 0o777)
    # TODO (Jannes): Remove original file once satisfied with transformations
    logging.info("Movie file successfully converted and stored.")
    return conv_fpath


    
