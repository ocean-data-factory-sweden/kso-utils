# base imports
from lib2to3.pytree import convert
import os
import cv2
import pandas as pd
from tqdm import tqdm
import difflib
import logging
import subprocess
import urllib

# util imports
import kso_utils.server_utils as server_utils
import kso_utils.project_utils as project_utils
import kso_utils.tutorials_utils as tutorials_utils


# Logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_fps_duration(movie_path: str):
    '''
    This function takes the path (or url) of a movie and returns its fps and duration information
    
    :param movie_path: a string containing the path (or url) where the movie of interest can be access from
    :return: Two integers, the fps and duration of the movie
    """
    '''
    cap = cv2.VideoCapture(movie_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
                    
    return fps, duration


def get_movie_path(f_path: str, db_info_dict: dict, project: project_utils.Project):
    '''
    Function to get the path (or url) of a movie

    :param f_path: string with the original path of a movie
    :param db_info_dict: a dictionary with the initial information of the project
    :param project: the project object
    :return: a string containing the path (or url) where the movie of interest can be access from

    '''
    # Get the project-specific server
    server = project.server
    
    if server == "AWS":
        # Extract the key from the orignal path of the movie
        movie_key = f_path.replace("%20"," ").split('/',3)[3]

        # Generate presigned url
        movie_url = db_info_dict['client'].generate_presigned_url('get_object', 
                                                            Params = {'Bucket': server_dict['bucket'], 
                                                                      'Key': movie_key}, 
                                                            ExpiresIn = 86400)
        return movie_url

    elif server == "SNIC":
        logging.error("Work still to be completed")
    
    else:
        return f_path

def get_movie_extensions():
    # Specify the formats of the movies to select
    movie_formats = tuple(['wmv', 'mpg', 'mov', 'avi', 'mp4', 'MOV', 'MP4'])
    return movie_formats


def standarise_movie_format(movie_path: str, movie_filename: str, gpu_available: bool = False):
    """
    This function reviews the movie metadata. If the movie is not in the correct format, frame rate or codec,
    it is converted using ffmpeg. 
    
    :param movie_path: The local path- or url to the movie file you want to convert
    :type movie_path: str
    :param movie_filename: The filename of the movie file you want to convert
    :type movie_path: str
    :param gpu_available: Boolean, whether or not a GPU is available
    :type gpu_available: bool
    """

    # Check movie format
    filename, ext = os.path.splitext(movie_filename)
    
    if not ext.lower() == "mp4":
        logging.info("Extension of", movie_filename ,"not supported.")
        # Set conversion to True
        convert_video = True

    # Check frame rate
    cap = cv2.VideoCapture(movie_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not float(fps).is_integer():
        logging.info("Variable frame rate of", movie_filename,"not supported.")
        # Set conversion to True
        convert_video = True

    # Check codec information
    h = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = chr(h&0xff) + chr((h>>8)&0xff) + chr((h>>16)&0xff) + chr((h>>24)&0xff)
    
    if not codec == "h264":
        logging.info("The codecs of", movie_filename, "are not supported (only h264 is supported).")
        # Set conversion to True
        convert_video = True

    # Check movie filesize in relation to its duration
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    duration_mins = duration/60
    
    # Check if the movie is accessible locally
    if os.path.exists(movie_path):
        # Store the size of the movie
        size = os.path.getsize(movie_path)

    # Check if the path to the movie is a url
    elif tutorials_utils.is_url(movie_path):
        # Store the size of the movie
        size = urllib.request.urlopen(movie_path).length
    
    else:
        logging.error("The path to", movie_path," is invalid")      
      
    sizeGB = size/(1024*1024*1024)
    size_duration = sizeGB/duration_mins

    if size_duration > 0.16:
        logging.info("File size of movie is too large (+5GB per 30 mins of footage).")
        # Set conversion to True
        convert_video = True
           
    # Start converting video if movie didn't pass any of the checks
    if convert_video:
        if gpu_available:
            print("gpu_available")
            new_mov_path = convert_video(movie_path, movie_filename, True)
        else:
            print("gpu_available not available")
            new_mov_path = convert_video(movie_path, movie_filename, False)

    else:
        logging.info(movie_filename, "format is standard.")
        

def convert_video(movie_path: str, movie_filename: str, gpu_available: bool = False):
    """
    It takes a movie file path and a boolean indicating whether a GPU is available, and returns a new
    movie file path.
    
    :param movie_path: The local path- or url to the movie file you want to convert
    :type movie_path: str
    :param movie_filename: The filename of the movie file you want to convert
    :type movie_path: str
    :param gpu_available: Boolean, whether or not a GPU is available
    :type gpu_available: bool
    :return: The path to the converted video file.
    """
    print("made it here")

    conv_filename = "conv_" + movie_filename
    
    # Check the movie is accessible locally
    if os.path.exists(movie_path):
      # Store the directory and filename of the movie
      movie_fpath = os.path.dirname(movie_path)
      conv_fpath = os.path.join(movie_fpath, conv_filename) 

    # Check if the path to the movie is a url
    elif tutorials_utils.is_url(movie_path):
      # Specify the directory to store the converted movie
      conv_fpath = os.path.join(conv_filename)

    else:
      logging.error("The path to", movie_path," is invalid")

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


    
