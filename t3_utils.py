# -*- coding: utf-8 -*-
# base imports
import os, shutil, ffmpeg
import pandas as pd
import numpy as np
import math
import datetime
import subprocess
import logging
import random
import difflib
from pathlib import Path

# widget imports
from tqdm import tqdm
from IPython.display import HTML, display, clear_output
from ipywidgets import interact, interactive, Layout
import ipywidgets as widgets
from panoptes_client import (
    SubjectSet,
    Subject,
    Project,
    Panoptes,
)

# util imports
import kso_utils.db_utils as db_utils
import kso_utils.server_utils as server_utils

# Logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


   
############################################################
######## Create some clip examples #########################
############################################################


# Select the movie you want
def select_movie(available_movies_df):

    # Get the list of available movies
    available_movies_tuple = tuple(sorted(available_movies_df.filename.unique()))
    
    # Widget to select the movie
    select_movie_widget = widgets.Dropdown(
                    options = available_movies_tuple,
                    description = "Movie of interest:",
                    ensure_option = True,
                    disabled = False,
                    layout = widgets.Layout(width='50%'),
                    style = {'description_width': 'initial'},
                )
    
    display(select_movie_widget)
    
    return select_movie_widget


def check_movie_uploaded(movie_i, db_info_dict):

    # Create connection to db
    conn = db_utils.create_connection(db_info_dict["db_path"])

    # Query info about the clip subjects uploaded to Zooniverse
    subjects_df = pd.read_sql_query("SELECT id, subject_type, filename, clip_start_time, clip_end_time, movie_id FROM subjects WHERE subject_type='clip'", conn)

    # Save the video filenames of the clips uploaded to Zooniverse 
    videos_uploaded = subjects_df.filename.unique()

    # Check if selected movie has already been uploaded
    already_uploaded = any(mv in movie_i for mv in videos_uploaded)

    if already_uploaded:
        clips_uploaded = subjects_df[subjects_df["filename"].str.contains(movie_i)]
        print(movie_i, "has clips already uploaded. The clips start and finish at:")
        print(clips_uploaded[["clip_start_time", "clip_end_time"]], sep = "\n")
    else:
        print(movie_i, "has not been uploaded to Zooniverse yet")
        
def select_clip_length():
    # Widget to record the length of the clips
    ClipLength_widget = widgets.Dropdown(
        options=[10,5],
        value=10,
        description="Length of clips:",
        style = {'description_width': 'initial'},
        ensure_option=True,
        disabled=False
    )  

    return ClipLength_widget


def select_random_clips(movie_i, db_info_dict):
  # Create connection to db
    conn = db_utils.create_connection(db_info_dict["db_path"])

    # Query info about the movie of interest
    movie_df = pd.read_sql_query(
        f"SELECT filename, duration, sampling_start, sampling_end FROM movies WHERE filename='{movie_i}'", conn)
    
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
                "random_clip_length": clip_length
        }

        print("The initial seconds of the examples will be:", *random_clips_info["clip_start_time"], sep = "\n")

        return random_clips_info


    # Select the number of clips to upload 
    clip_length_number = interactive(n_random_clips, 
                                     clip_length = select_clip_length(),
                                     n_clips = widgets.IntSlider(
                                          value=3,
                                          min=1,
                                          max=5,
                                          step=1,
                                          description='Number of random clips:',
                                          disabled=False,
                                          layout=Layout(width='40%'),
                                          style = {'description_width': 'initial'})
                                     )

                                   
    display(clip_length_number)
    
    return clip_length_number  


# Function to extract the videos 
def extract_random_clips(clips_start_time, clip_length, movie_i, movie_path, clips_folder): 
    random_clips = []
    
    # Create the information for each clip and extract it (printing a progress bar) 
    for start_time_i in tqdm(clips_start_time):
        # Create the filename and path of the clip
        output_clip_name = movie_i + "_clip_" + str(start_time_i) + "_" + str(clip_length) + ".mp4"
        output_clip_path = clips_folder + os.sep + output_clip_name
        
        # Print statements to check all good
        print("start_time_i", str(start_time_i))
        print("clip_length", str(clip_length))
        print("output_clip_path", str(output_clip_path))
        print("movie_path", str(movie_path))
        
        # Add the path of the clip to the list
        random_clips = random_clips + [output_clip_path]
        
        # Extract the clip
        if not os.path.exists(output_clip_path):
            subprocess.call(["ffmpeg", 
                             "-ss", str(start_time_i), 
                             "-t", str(clip_length), 
                             "-i", str(movie_path), 
                             "-c", "copy", 
                             "-an",#removes the audio
                             "-force_key_frames", "1",
                             str(output_clip_path)])

            os.chmod(output_clip_path, 0o755)
    print("Clips extracted successfully")
    
    return random_clips
    
def create_example_clips(movie_i, movie_path, db_info_dict, project, clip_selection):
    
    # Specify the starting seconds and length of the example clips
    clips_start_time = clip_selection.result["clip_start_time"]
    clip_length = clip_selection.result["random_clip_length"]

    # Get project-specific server info
    server = project.server
    
    # Specify the temp folder to host the clips
    if server == "SNIC":
        clips_folder = "/cephyr/NOBACKUP/groups/snic2021-6-9/tmp_dir/" + movie_i + "_clips"
    else:
        clips_folder = movie_i + "_clips"

    # Create the folder to store the videos if not exist
    if not os.path.exists(clips_folder):
        os.mkdir(clips_folder)

    # Extract the clips and store them in the folder
    random_clips = extract_random_clips(clips_start_time = clips_start_time, 
                   clip_length = clip_length,
                   movie_i = movie_i,
                   movie_path = movie_path, 
                   clips_folder = clips_folder)

    return random_clips    


def check_clip_size(clips_list):
    
    # Get list of files with size
    files_with_size = [ (file_path, os.stat(file_path).st_size) 
                        for file_path in clips_list]

    df = pd.DataFrame(files_with_size, columns=["File_path","Size"])

    #Change bytes to MB
    df['Size'] = df['Size']/1000000
    
    if df['Size'].ge(8).any():
        print("Clips are too large (over 8 MB) to be uploaded to Zooniverse. Compress them!")
        return df
    else:
        print("Clips are a good size (below 8 MB). Ready to be uploaded to Zooniverse")
        return df



class clip_modification_widget(widgets.VBox):

    def __init__(self):
        '''
        The function creates a widget that allows the user to select which modifications to run
        '''
        self.widget_count = widgets.IntText(description='Number of modifications:',
                                            display="flex",
                                            flex_flow="column",
                                            align_items="stretch",
                                            style={"description_width": "initial"})
        self.bool_widget_holder = widgets.HBox(layout=widgets.Layout(width='100%',
                                                                     display='inline-flex',
                                                                     flex_flow='row wrap'))
        children = [
            self.widget_count,
            self.bool_widget_holder,
        ]
        self.widget_count.observe(self._add_bool_widgets, names=['value'])
        super().__init__(children=children)

    def _add_bool_widgets(self, widg):
        num_bools = widg['new']
        new_widgets = []
        for _ in range(num_bools):
            new_widget = select_modification()
            for wdgt in [new_widget]:
                wdgt.description = wdgt.description + f" #{_}"
            new_widgets.extend([new_widget])
        self.bool_widget_holder.children = tuple(new_widgets)

    @property
    def checks(self):
        return {
            w.description: w.value
            for w in self.bool_widget_holder.children
        }


def select_modification():
    # Widget to select the clip modification

    clip_modifications = {"Color_correction": {"filter":
                            ".filter('curves', '0/0 0.396/0.67 1/1', \
                                        '0/0 0.525/0.451 1/1', \
                                        '0/0 0.459/0.517 1/1')"} 
                            #borrowed from https://www.element84.com/blog/color-correction-in-space-and-at-sea
                            , "Zoo_low_compression": {
                            "crf": "25",
                            }, "Zoo_medium_compression": {
                            "crf": "27",
                            }, "Zoo_high_compression": {
                            "crf": "30",
                            }, "Blur_sensitive_info": { "filter":
                            ".drawbox(0, 0, 'iw', 'ih*(15/100)', color='black' \
                            ,thickness='fill').drawbox(0, 'ih*(95/100)', \
                            'iw', 'ih*(15/100)', color='black', thickness='fill')",
                            "None": {}}}
    
    select_modification_widget = widgets.Dropdown(
                    options=[(a,b) for a,b in clip_modifications.items()],
                    description="Select modification:",
                    ensure_option=True,
                    disabled=False,
                    style = {'description_width': 'initial'}
                )
    
    #display(select_modification_widget)
    return select_modification_widget


def gpu_select():
    
    def gpu_output(gpu_option):
        if gpu_option == "No GPU":
            print("You are set to start the modifications")
            # Set GPU argument
            gpu_available = False
            return gpu_available
        
        if gpu_option == "Colab GPU":
            print("Installing the requirements for GPU video modification")
            # Install ffmpeg with GPU version
#             !git clone https://github.com/rokibulislaam/colab-ffmpeg-cuda.git
#             !cp -r ./colab-ffmpeg-cuda/bin/. /usr/bin/
            
            # Set GPU argument
            gpu_available = True
            return gpu_available
        
        if gpu_option == "Other GPU":
            # Set GPU argument
            gpu_available = True
            return gpu_available
            
        

    # Select the gpu availability
    gpu_output_interact = interactive(gpu_output, 
                                      gpu_option = widgets.RadioButtons(
                                        options = ['No GPU', 'Colab GPU', 'Other GPU'],
                                        value = 'No GPU', 
                                        description = 'Select GPU availability:',
                                        disabled = False
                                        )
                                    )
    
                                   
    display(gpu_output_interact)
    
    return gpu_output_interact


def modify_clips(clips_list, modification_details, mod_clips_folder, gpu_available):
    
    modified_clips = []
    
    # Create the information for each clip and modify it (printing a progress bar) 
    for clip_i in tqdm(clips_list):
        # Create the filename and path of the modified clip
        output_clip_name = "modified_" + os.path.basename(clip_i)
        output_clip_path = mod_clips_folder + os.sep + output_clip_name
        
        print(output_clip_path)
        
        if not os.path.exists("output_clip_path"):
            if gpu_available:
                subprocess.call(["ffmpeg",
                                 "-hwaccel", "cuda",
                                 "-hwaccel_output_format", "cuda",
                                 "-i", clip_i, 
                                 "-c:v", "h264_nvenc",
                                 output_clip_path])
                
            else:
                # Set up input prompt
                init_prompt = f"ffmpeg.input('{clip_i}')"
                full_prompt = init_prompt
                    
                # Set up modification
                for transform in modification_details.values():
                    if "filter" in transform:
                        mod_prompt = transform['filter']
                        full_prompt += mod_prompt
                    
                    # Setup output prompt
                    crf_value = [transform["crf"] if "crf" in transform else None for transform in modification_details.values()]
                    crf_value = [i for i in crf_value if i is not None]

                    if len(crf_value) > 0:
                        crf_prompt = str(max([int(i) for i in crf_value]))
                        full_prompt += f".output('{output_clip_path}', crf={crf_prompt}, preset='veryfast', pix_fmt='yuv420p', vcodec='libx264')"
                    else:
                        full_prompt += f".output('{output_clip_path}', crf=20, pix_fmt='yuv420p', vcodec='libx264')"
                    
                    # Run the modification
                    try:
                        eval(full_prompt).run(capture_stdout=True, capture_stderr=True)
                        os.chmod(output_clip_path, 0o755)
                    except ffmpeg.Error as e:
                        print('stdout:', e.stdout.decode('utf8'))
                        print('stderr:', e.stderr.decode('utf8'))
                        raise e
            
            # Add the path of the clip to the list
            modified_clips = modified_clips + [output_clip_path]
    print("Clips modified successfully")
    
    return modified_clips
        
    
def create_modified_clips(clips_list, movie_i, modification_details, project, gpu_available):

    # Get project-specific server info
    server = project.server

    # Specify the folder to host the modified clips
    if server == "SNIC":
        mod_clips_folder = "/cephyr/NOBACKUP/groups/snic2021-6-9/tmp_dir/"+"modified_" + movie_i + "_clips"
    else:
        mod_clips_folder = "modified_" + movie_i +"_clips"
    
    # Remove existing modified clips
    if os.path.exists(mod_clips_folder):
        shutil.rmtree(mod_clips_folder)
    
    if len(modification_details.values()) > 0:
        
        # Create the folder to store the videos if not exist
        if not os.path.exists(mod_clips_folder):
            os.mkdir(mod_clips_folder)
            
        # Extract the clips and store them in the folder
        modified_clips = modify_clips(clips_list = clips_list, 
                                    modification_details = str(modification_details),
                                    mod_clips_folder = mod_clips_folder,
                                     gpu_available = gpu_available)

        return modified_clips 
    else:
        print("No modification selected")
    
    
# Display the clips side-by-side
def view_clips(modified_clips, random_clip_path):
    
    # Get the path of the modified clip selected
    modified_clip_name = os.path.basename(random_clip_path).replace("modified_", "")
    modified_clip_path = (x for x in modified_clips if os.path.basename(x) == modified_clip_name)

    # Open original video
    vid1 = open(random_clip_path,'rb').read()
    wi1 = widgets.Video(value = vid1, format = extension, 
                        width = 400, height = 500)
    
    # Open modified video
    vid2 = open(modified_clip_path,'rb').read()
    wi2 = widgets.Video(value = vid2, format = extension, 
                        width = 400, height = 500)
    
    # Display videos side-by-side
    a = [wi1, wi2]
    wid = widgets.HBox(a)

    return wid

def compare_clips(random_clips, modified_clips):

    # Add "no movie" option to prevent conflicts
    random_clips = np.append(random_clips,"0 No movie")
    
    clip_path_widget = widgets.Dropdown(
                    options=tuple(random_clips),
                    description="Select original clip:",
                    ensure_option=True,
                    disabled=False,
                    layout=Layout(width='50%'),
                    style = {'description_width': 'initial'}
                )
    
    main_out = widgets.Output()
    display(clip_path_widget, main_out)
    
    # Display the original and modified clips
    def on_change(change):
        with main_out:
            clear_output()
            if change["new"]=="0 No movie":
                print("It is OK to modify the clips again")
            else:
                a = view_clips(modified_clips, change["new"])
                display(a)
                
                
    clip_path_widget.observe(on_change, names='value')        
        
        
        
############################################################
######## Create the clips to upload to Zooniverse ##########
############################################################

def select_clip_n_len(movie_i, db_info_dict):
    
    # Create connection to db
    conn = db_utils.create_connection(db_info_dict["db_path"])

    # Query info about the movie of interest
    movie_df = pd.read_sql_query(
        f"SELECT filename, duration, sampling_start, sampling_end FROM movies WHERE filename='{movie_i}'", conn)
    
    # Display in hours, minutes and seconds
    def to_clips(clip_length, clips_range):

        # Calculate the number of clips
        clips = int((clips_range[1]-clips_range[0])/clip_length)

        print("Number of clips to upload:", clips)

        return clips


    # Select the number of clips to upload 
    clip_length_number = interactive(to_clips, 
                              clip_length = select_clip_length(),
                             clips_range = widgets.IntRangeSlider(value=[movie_df.sampling_start.values,
                                                                         movie_df.sampling_end.values],
                                                                  min=0,
                                                                  max=int(movie_df.duration.values),
                                                                  step=1,
                                                                  description='Range in seconds:',
                                                                  style = {'description_width': 'initial'},
                                                                  layout=widgets.Layout(width='90%')
                                                                 ))

                                   
    display(clip_length_number)
    
    return clip_length_number        

def review_clip_selection(clip_selection, movie_i, clip_modification):
    start_trim = clip_selection.kwargs['clips_range'][0]
    end_trim = clip_selection.kwargs['clips_range'][1]

    # Review the clips that will be created
    print("You are about to create", round(clip_selection.result), 
          "clips from", movie_i)
    print("starting at", datetime.timedelta(seconds=start_trim), 
          "and ending at", datetime.timedelta(seconds=end_trim))
    print("The modification selected is", clip_modification)


# Func to expand seconds
def expand_list(df, list_column, new_column):
    lens_of_lists = df[list_column].apply(len)
    origin_rows = range(df.shape[0])
    destination_rows = np.repeat(origin_rows, lens_of_lists)
    non_list_cols = [idx for idx, col in enumerate(df.columns) if col != list_column]
    expanded_df = df.iloc[destination_rows, non_list_cols].copy()
    expanded_df[new_column] = [item for items in df[list_column] for item in items]
    expanded_df.reset_index(inplace=True, drop=True)
    return expanded_df

# Function to extract the videos 
def extract_clips(df, movie_path, clip_length, clip_modification, gpu_available): 
    # Read each movie and extract the clips (printing a progress bar) 
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if not os.path.exists(row['clip_path']):
            if gpu_available:
                subprocess.call(["ffmpeg",
                                 "-hwaccel", "cuda",
                                 "-hwaccel_output_format", "cuda",                                 
                                 "-ss", str(row['upl_seconds']), 
                                 "-t", str(clip_length), 
                                 "-i", movie_path,
                                 "-an",#removes the audio
                                 "-c:v", "h264_nvenc",
                                 str(row['clip_path'])])
                os.chmod(row['clip_path'], 0o755)
            else:
                # Set up input prompt
                init_prompt = f"ffmpeg.input('{movie_path}')"
                full_prompt = init_prompt
                    
                # Set up modification
                for transform in modification_details.values():
                    if "filter" in transform:
                        mod_prompt = transform['filter']
                        full_prompt += mod_prompt
                    
                    # Setup output prompt
                    crf_value = [transform["crf"] if "crf" in transform else None for transform in modification_details.values()]
                    crf_value = [i for i in crf_value if i is not None]

                    if len(crf_value) > 0:
                        crf_prompt = str(max([int(i) for i in crf_value]))
                        full_prompt += f".output('{str(row['clip_path'])}', crf={crf_prompt}, ss={str(row['upl_seconds'])}, t={str(clip_length)}, preset='veryfast', pix_fmt='yuv420p', vcodec='libx264')"
                    else:
                        full_prompt += f".output('{str(row['clip_path'])}', ss={str(row['upl_seconds'])}, t={str(clip_length)}, crf=20, pix_fmt='yuv420p', vcodec='libx264')"
                    
                    # Run the modification
                    try:
                        eval(full_prompt).run(capture_stdout=True, capture_stderr=True)
                        os.chmod(str(row['clip_path']), 0o755)
                    except ffmpeg.Error as e:
                        print('stdout:', e.stdout.decode('utf8'))
                        print('stderr:', e.stderr.decode('utf8'))
                        raise e
    
                
                
        print("clips extracted successfully")
                
    
def create_clips(available_movies_df, movie_i, movie_path, db_info_dict, clip_selection, project, clip_modification, gpu_available):
        
    # Filter the df for the movie of interest
    movie_i_df = available_movies_df[available_movies_df['filename']==movie_i].reset_index(drop=True)

    # Calculate the max number of clips available
    clip_length = clip_selection.kwargs['clip_length']
    clip_numbers = clip_selection.result
    start_trim = clip_selection.kwargs['clips_range'][0]
    end_trim = clip_selection.kwargs['clips_range'][1]

    # Calculate all the seconds for the new clips to start
    movie_i_df["seconds"] = [
        list(range(start_trim, int(math.floor(end_trim / clip_length) * clip_length), clip_length))
    ]

    # Reshape the dataframe with the seconds for the new clips to start on the rows
    potential_start_df = expand_list(movie_i_df, "seconds", "upl_seconds")

    # Specify the length of the clips
    potential_start_df["clip_length"] = clip_length

    if not clip_numbers==potential_start_df.shape[0]:
        print("There was an issue estimating the starting seconds for the", clip_numbers, "clips")

    # Get project-specific server info
    server = project.server
    
    # Specify the temp folder to host the clips
    if server == "SNIC":
        clips_folder = "/cephyr/NOBACKUP/groups/snic2021-6-9/tmp_dir/" + movie_i + "_zooniverseclips"
    else:
        clips_folder = movie_i+"_zooniverseclips"

    # Set the filename of the clips
    potential_start_df["clip_filename"] = movie_i + "_clip_" + potential_start_df["upl_seconds"].astype(str) + "_" + str(clip_length) + ".mp4"

    # Set the path of the clips
    potential_start_df["clip_path"] = clips_folder + os.sep + potential_start_df["clip_filename"]

    # Create the folder to store the videos if not exist
    if not os.path.exists(clips_folder):
        os.mkdir(clips_folder)

    # Extract the videos and store them in the folder
    extract_clips(potential_start_df, movie_path, clip_length, clip_modification, gpu_available)

    return potential_start_df      


def set_zoo_metadata(df, project, db_info_dict):
    
    # Create connection to db
    conn = db_utils.create_connection(db_info_dict["db_path"])

    # Query info about the movie of interest
    sitesdf = pd.read_sql_query(
        f"SELECT * FROM sites", conn)
        
    # Combine site info to the df
    if "site_id" in df.columns:
        upload_to_zoo = df.merge(sitesdf, left_on="site_id", right_on="id")
        sitename = upload_to_zoo["siteName"].unique()[0]
    else:
        raise ValueError("Sites table empty. Perhaps try to rebuild the initial db.")
        
    # Rename columns to match schema names
    # (fields that begin with “#” or “//” will never be shown to volunteers)
    # (fields that begin with "!" will only be available for volunteers on the Talk section, after classification)
    upload_to_zoo = upload_to_zoo.rename(columns={
            "created_on": "#created_on",
            "clip_length": "#clip_length",
            "filename": "#VideoFilename",
            "clip_modification_details": "#clip_modification_details",
            "siteName": "#siteName"
            })

    # Set clip path if no modification
    if "modif_clip_path" in upload_to_zoo.columns and "no_modification" not in upload_to_zoo["modif_clip_path"].values:
        upload_to_zoo["modif_clip_path"] = upload_to_zoo["clip_path"]
    
    # Convert datetime to string to avoid JSON seriazible issues
    upload_to_zoo['#created_on'] = upload_to_zoo['#created_on'].astype(str)
    created_on = upload_to_zoo['#created_on'].unique()[0]
        
    # Select only relevant columns
    upload_to_zoo = upload_to_zoo[
        [
            "modif_clip_path",
            "upl_seconds",
            "#clip_length",
            "#created_on",
            "#VideoFilename",
            "#siteName",
            "#clip_modification_details"
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
        
        # Include movie info to the sites df
        sitesdf = sitesdf.merge(moviesdf, on="SiteID")
        
        # Rename columns to match schema names
        sitesdf = sitesdf.rename(columns={
            "LinkToMarineReserve": "!LinkToMarineReserve",
            "SiteID": "#SiteID",
            })
                                           
        # Select only relevant columns
        sitesdf = sitesdf[
            [
                "!LinkToMarineReserve",
                "#SiteID",
                "ProtectionStatus"
            ]
        ]
        
        # Include site info to the df
        upload_to_zoo = upload_to_zoo.merge(sitesdf, left_on="#siteName",
                                            right_on="#SiteID")
        
    
    if project.Project_name == "Koster_Seafloor_Obs":
        
        # Read sites csv as pd
        sitesdf = pd.read_csv(db_info_dict["local_sites_csv"])

        # Rename columns to match schema names
        sitesdf = sitesdf.rename(columns={
            "decimalLatitude": "#decimalLatitude",
            "decimalLongitude": "#decimalLongitude",
            "geodeticDatum": "#geodeticDatum",
            "countryCode": "#countryCode",
            })
                                           
        # Select only relevant columns
        sitesdf = sitesdf[
            [
                "siteName",
                "#decimalLatitude",
                "#decimalLongitude",
                "#geodeticDatum",
                "#countryCode"
            ]
        ]
        
        # Include site info to the df
        upload_to_zoo = upload_to_zoo.merge(sitesdf, left_on="#siteName",
                                            right_on="siteName")
        
    # Prevent NANs on any column
    if upload_to_zoo.isnull().values.any():
        print("The following columns have NAN values", 
              upload_to_zoo.columns[upload_to_zoo.isna().any()].tolist())
        
    return upload_to_zoo, sitename, created_on

def upload_clips_to_zooniverse(upload_to_zoo, sitename, created_on, project):
    
    # Estimate the number of clips
    n_clips = upload_to_zoo.shape[0]
    
    # Create a new subject set to host the clips
    subject_set = SubjectSet()

    subject_set_name = "clips_" + sitename + "_" + str(int(n_clips)) + "_" + created_on
    subject_set.links.project = project
    subject_set.display_name = subject_set_name

    subject_set.save()

    print(subject_set_name, "subject set created")

    # Save the df as the subject metadata
    subject_metadata = upload_to_zoo.set_index('modif_clip_path').to_dict('index')

    # Upload the clips to Zooniverse (with metadata)
    new_subjects = []

    print("uploading subjects to Zooniverse")
    for modif_clip_path, metadata in tqdm(subject_metadata.items(), total=len(subject_metadata)):
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

    print("Subjects uploaded to Zooniverse")        
        
        
