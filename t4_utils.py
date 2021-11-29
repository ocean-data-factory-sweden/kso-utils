#t4 utils
import os, shutil
import pandas as pd
import numpy as np
import math
import datetime
import subprocess

from tqdm import tqdm
from IPython.display import HTML, display, update_display, clear_output
from ipywidgets import interact, interactive, Layout
from utils.zooniverse_utils import auth_session
import ipywidgets as widgets

import utils.db_utils as db_utils
import utils.zooniverse_utils as zooniverse_utils
import utils.movie_utils as movie_utils
import utils.server_utils as server_utils
import utils.tutorials_utils as tutorials_utils
import utils.koster_utils as koster_utils
from panoptes_client import (
    SubjectSet,
    Subject,
    Project,
    Panoptes,
)
   
def retrieve_movie_info_from_server(project_name, db_info_dict):
    
    if project_name == "Spyfish_Aotearoa":
        # Specify the bucket
        bucket_i = 'marine-buv'

        # Retrieve info from the bucket
        server_df = server_utils.get_matching_s3_keys(client = db_info_dict["client"], 
                                                         bucket = bucket_i, 
                                                         suffix = movie_utils.get_movie_extensions())
        server_df.rename({"Key": "spath"})
    
    if project_name == "Koster_Seafloor_Obs":
        server_df = server_utils.get_koster_movies(client = db_info_dict["client"])
        server_df["spath"] = server_df["spath"].apply(koster_utils.unswedify)
    else:
        raise ValueError("The project you selected is not currently supported.")
    
    
    # Create connection to db
    conn = db_utils.create_connection(db_info_dict["db_path"])

    # Query info about the movie of interest
    movies_df = pd.read_sql_query(f"SELECT * FROM movies", conn)

    # Missing info for files in the "buv-zooniverse-uploads"
    movies_df = movies_df.merge(server_df["spath"], 
                                left_on=['fpath'],
                                right_on=['spath'], 
                                how='left', 
                                indicator=True)

    # Check that movies can be mapped
    movies_df['exists'] = np.where(movies_df["_merge"]=="left_only", False, True)

    # Drop _merge columns to match sql schema
    movies_df = movies_df.drop("_merge", axis=1)
    
    # Select only those that can be mapped
    available_movies_df = movies_df[movies_df['exists']]
    
    # Create a filename with ext column
    available_movies_df["filename_ext"] = available_movies_df["fpath"].str.split("/").str[-1]

    print(available_movies_df.shape[0], "movies are mapped from the server")
    
    return available_movies_df

# Select the movie you want to upload to Zooniverse
def movie_to_upload(available_movies_df):

    # Widget to select the movie
    movie_to_upload_widget = widgets.Combobox(
                    options=tuple(available_movies_df.filename.unique()),
                    description="Movie to upload:",
                    ensure_option=True,
                    disabled=False,
                    layout=Layout(width='50%'),
                    style = {'description_width': 'initial'},
                )
    
    
    display(movie_to_upload_widget)
    return movie_to_upload_widget


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
        
        
def select_clip_n_len(movie_i, db_info_dict):
    
    # Create connection to db
    conn = db_utils.create_connection(db_info_dict["db_path"])

    # Query info about the movie of interest
    movie_df = pd.read_sql_query(
        f"SELECT filename, duration, survey_start, survey_end FROM movies WHERE filename='{movie_i}'", conn)
    
    # Display in hours, minutes and seconds
    def to_clips(clip_length, clips_range):

        # Calculate the number of clips
        clips = (clips_range[1]-clips_range[0])/clip_length

        print("Number of clips to upload:", clips)

        return clips


    # Select the number of clips to upload 
    clip_length_number = interactive(to_clips, 
                              clip_length = widgets.Dropdown(
                                 options=[10,5],
                                 value=10,
                                 description="Length of clips:",
                                 ensure_option=True,
                                 disabled=False,),
                             clips_range = widgets.IntRangeSlider(value=[movie_df.survey_start.values,
                                                                         movie_df.survey_end.values],
                                                                  min=0,
                                                                  max=movie_df.duration.values,
                                                                  step=1,
                                                                  description='Range in seconds:',
                                                                 ))

                                   
    display(clip_length_number)
    
    return clip_length_number        
        
def review_clip_selection(clip_selection, movie_i):
    start_trim = clip_selection.kwargs['clips_range'][0]
    end_trim = clip_selection.kwargs['clips_range'][1]

    # Review the clips that will be created
    print("You are about to create", round(clip_selection.result), 
          "clips from", movie_i)
    print("starting at", datetime.timedelta(seconds=start_trim), 
          "and ending at", datetime.timedelta(seconds=end_trim))


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
def extract_clips(df, clip_length): 
    # Read each movie and extract the clips (printing a progress bar) 
    for index, row in tqdm(df.iterrows(), total=df.shape[0]): 
        if not os.path.exists(row['clip_path']):
            subprocess.call(["ffmpeg", 
                             "-ss", str(row['upl_seconds']), 
                             "-t", str(clip_length), 
                             "-i", str(row['filename_ext']), 
                             "-c", "copy", 
                             "-an",#removes the audio
                             "-force_key_frames", "1",
                             str(row['clip_path'])])

    print("clips extracted successfully")
    
    
def create_clips(available_movies_df, movie_i, db_info_dict, clip_selection, project_name):
        
        # Calculate the max number of clips available
        clip_length = clip_selection.kwargs['clip_length']
        clip_numbers = clip_selection.result
        start_trim = clip_selection.kwargs['clips_range'][0]
        end_trim = clip_selection.kwargs['clips_range'][1]

        # Filter the df for the movie of interest
        movie_i_df = available_movies_df[available_movies_df['filename']==movie_i].reset_index(drop=True)
        
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
            
        # Download the movie locally from the server
        if project_name == "Spyfish_Aotearoa":

            bucket_i = "marine-buv"

            if not os.path.exists(movie_i_df.filename_ext[0]):
                # Download the movie of interest
                server_utils.download_object_from_s3(
                                db_info_dict["client"],
                                bucket=bucket_i,
                                key=movie_i_df.Key.unique()[0],
                                filename=movie_i_df.filename_ext[0],
                )        

        # Specify the temp folder to host the clips
        clips_folder = movie_i+"_clips"

        # Set the filename of the clips
        potential_start_df["clip_filename"] = movie_i + "_clip_" + potential_start_df["upl_seconds"].astype(str) + "_" + str(clip_length) + ".mp4"

        # Set the path of the clips
        potential_start_df["clip_path"] = clips_folder+ "/" + potential_start_df["clip_filename"]

        # Create the folder to store the videos if not exist
        if not os.path.exists(clips_folder):
            os.mkdir(clips_folder)

        # Extract the videos and store them in the folder
        extract_clips(potential_start_df, clip_length)
        
        return potential_start_df    

    
def check_clip_size(clip_paths):
    
    # Get list of files with size
    files_with_size = [ (file_path, os.stat(file_path).st_size) 
                        for file_path in clip_paths]

    df = pd.DataFrame(files_with_size, columns=["File_path","Size"])

    #Change bytes to MB
    df['Size'] = df['Size']/1000000
    
    if df['Size'].ge(8).any():
        print("Clips are too large (over 8 MB) to be uploaded to Zooniverse. Compress them!")
        return df
    else:
        print("Clips are a good size (below 8 MB). Ready to be uploaded to Zooniverse")
        return df
    

def select_modification():
    # Widget to select the clip modification
    select_modification_widget = widgets.Combobox(
                    options=["None","Color_correction","Zoo_compression", "Blur_sensitive_info"],
                    description="Select clip modification:",
                    ensure_option=True,
                    disabled=False,
                )
    
    
    display(select_modification_widget)
    return select_modification_widget


def modify_clips(clips_to_upload_df, movie_i, clip_modification, modification_details):

    # Specify the folder to host the modified clips
    mod_clips_folder = "modified_" + movie_i +"_clips"
    
    # Specify the path of the modified clips
    clips_to_upload_df["modif_clip_path"] = mod_clips_folder + "\modified" + clips_to_upload_df.clip_filename
    
    # Remove existing modified clips
    if os.path.exists(mod_clips_folder):
        shutil.rmtree(mod_clips_folder)

    if not clip_modification=="None":
        
        # Save the modification details to include as subject metadata
        clips_to_upload_df["clip_modification_details"] = str(modification_details)
        
        # Create the folder to store the videos if not exist
        if not os.path.exists(mod_clips_folder):
            os.mkdir(mod_clips_folder)

        #### Modify the clips###
        # Read each clip and modify them (printing a progress bar) 
        for index, row in tqdm(clips_to_upload_df.iterrows(), total=clips_to_upload_df.shape[0]): 
            if not os.path.exists(row['modif_clip_path']):
                if "-vf" in modification_details:
                    subprocess.call(["ffmpeg",
                                     "-i", str(row['clip_path']),
                                     "-c:v", modification_details["-c:v"],
                                     "-vf", modification_details["-vf"],
                                     "-crf", modification_details["-crf"],
                                     "-pix_fmt", modification_details["-pix_fmt"],
                                     "-preset", modification_details["-preset"],
                                     str(row['modif_clip_path'])])
                        
                else:
                    subprocess.call(["ffmpeg",
                                     "-i", str(row['clip_path']),
                                     "-c:v", modification_details["-c:v"],
                                     "-crf", modification_details["-crf"],
                                     "-pix_fmt", modification_details["-pix_fmt"],
                                     "-preset", modification_details["-preset"],
                                     str(row['modif_clip_path'])])


        print("Clips modified successfully")
            
        
        return clips_to_upload_df
    
    else:
        
        # Save the modification details to include as subject metadata
        clips_to_upload_df["modif_clip_path"] = "no_modification"
        
        return clips_to_upload_df
    
    
def compare_clips(df):

    # Save the paths of the clips
    original_clip_paths = df["clip_path"].unique()
    
    # Add "no movie" option to prevent conflicts
    original_clip_paths = np.append(original_clip_paths,"0 No movie")
    
    clip_path_widget = widgets.Combobox(
                    options=tuple(np.sort(original_clip_paths)),
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
                a = view_clips(df, change["new"])
                display(a)
                
                
    clip_path_widget.observe(on_change, names='value')
    
    
# Display the clips using html
def view_clips(df, movie_path):
    
    # Get path of the modified clip selected
    modified_clip_path = df[df["clip_path"]==movie_path].modif_clip_path.values[0]
        
    html_code = f"""
        <html>
        <div style="display: flex; justify-content: space-around">
        <div>
          <video width=400 controls>
          <source src={movie_path} type="video/mp4">
        </video>
        </div>
        <div>
          <video width=400 controls>
          <source src={modified_clip_path} type="video/mp4">
        </video>
        </div>
        </html>"""
    
   
    return HTML(html_code)


def set_zoo_metadata(df, project_name, db_info_dict):
    
    # Get the site information
    sitename = df.siteName.unique()[0]

    # Create connection to db
    conn = db_utils.create_connection(db_info_dict["db_path"])

    # Query info about the movie of interest
    sitesdf = pd.read_sql_query(
        f"SELECT * FROM sites", conn)
        
    # Combine site info to the df
    upload_to_zoo = df.merge(sitesdf, on="siteName")
        
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
    
    # Convert datetime to string to avoid JSON seriazible issues
    upload_to_zoo['#created_on'] = upload_to_zoo['#created_on'].astype(str)
        
    # Select only relevant columns
    upload_to_zoo = upload_to_zoo[
        [
            "modif_clip_path",
            "upl_seconds",
            "#clip_length",
            "#created_on"
            "#VideoFilename",
            "#siteName",
            "#clip_modification_details"
        ]
    ]

    # Add information about the type of subject
    upload_to_zoo["Subject_type"] = "clip"
        
    # Add spyfish-specific info
    if project_name == "Spyfish_Aotearoa":
        
        # Read sites csv as pd
        sitesdf = pd.read_csv(db_info_dict["sites_csv"])

        # Rename columns to match schema names
        sitesdf = sitesdf.rename(columns={
            "Marine Reserve": "!LinkToMarineReserve",
            "Waypoint": "#SiteID",
            "Protection Status": "ProtectionStatus",
            "Depth (m)": "Depth",
            })
                                           
        # Select only relevant columns
        sitesdf = sitesdf[
            [
                "siteName",
                "!LinkToMarineReserve",
                "#SiteID",
                "ProtectionStatus",
                "Depth"
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

    subject_set_name = str(int(n_clips)) + "_clips" + "_" + sitename + created_on
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
        subject = Subject()
        
        
        subject.links.project = project
        subject.add_location(modif_clip_path)
        
        print(modif_clip_path)
        subject.metadata.update(metadata)
        
        print(metadata)
        subject.save()
        print("subject saved")
        new_subjects.append(subject)

    # Upload videos
    subject_set.add(new_subjects)

    print("Subjects uploaded to Zooniverse")
    
# def choose_movies(db_path):
    
#     # Connect to db
#     conn = db_utils.create_connection(db_path)
    
#     # Select all movies
#     movies_df = pd.read_sql_query(
#         f"SELECT filename, fpath FROM movies",
#         conn,
#     )
    
#     # Select only videos that can be mapped
#     available_movies_df = movies_df[movies_df['fpath'].map(os.path.isfile)]
    
#     ###### Select movies ####
#     # Display the movies available to upload
#     movie_selection = widgets.Combobox(
#         options = available_movies_df.filename.unique().tolist(),
#         description = 'Movie:',
#     )
    
#     ###### Select clip length ##########
#     # Display the length available
#     clip_length = widgets.RadioButtons(
#         options = [5,10],
#         value = 10,
#         description = 'Clip length (seconds):',
#     )
    
#     display(movie_selection, clip_length)
    
#     return movie_selection, clip_length

# def choose_clips(movie_selection, clip_length, db_path):
    
#     # Connect to db
#     conn = db_utils.create_connection(db_path)
    
#     # Select the movie to upload
#     movie_df = pd.read_sql_query(
#         f"SELECT id, filename, fps, survey_start, survey_end FROM movies WHERE movies.filename='{movie_selection}'",
#         conn,
#     )
    
#     print(movie_df.id.values)
    
#     # Get information of clips uploaded
#     uploaded_clips_df = pd.read_sql_query(
#         f"SELECT movie_id, clip_start_time, clip_end_time FROM subjects WHERE subjects.subject_type='clip' AND subjects.movie_id={movie_df.id.values}",
#         conn,
#     )

#     # Calculate the time when the new clips shouldn't start to avoid duplication (min=0)
#     uploaded_clips_df["clip_start_time"] = (
#         uploaded_clips_df["clip_start_time"] - clip_length
#     ).clip(lower=0)

#     # Calculate all the seconds when the new clips shouldn't start
#     uploaded_clips_df["seconds"] = [
#         list(range(i, j + 1))
#         for i, j in uploaded_clips_df[["clip_start_time", "clip_end_time"]].values
#     ]

#     # Reshape the dataframe of the seconds when the new clips shouldn't start
#     uploaded_start = expand_list(uploaded_clips_df, "seconds", "upl_seconds")[
#         ["movie_id", "upl_seconds"]
#     ]

#     # Exclude starting times of clips that have already been uploaded
#     potential_clips_df = (
#         pd.merge(
#             potential_start_df,
#             uploaded_start,
#             how="left",
#             left_on=["movie_id", "pot_seconds"],
#             right_on=["movie_id", "upl_seconds"],
#             indicator=True,
#         )
#         .query('_merge == "left_only"')
#         .drop(columns=["_merge"])
#     )
    
#     # Combine the flatten metadata with the subjects df
#     subj_df = pd.concat([subj_df, meta_df], axis=1)

#     # Filter clip subjects
#     subj_df = subj_df[subj_df['Subject_type']=="clip"]

#     # Create a dictionary with the right types of columns
#     subj_df = {
#     'subject_id': subj_df['subject_id'].astype(int),
#     'upl_seconds': subj_df['upl_seconds'].astype(int),
#     'clip_length': subj_df['#clip_length'].astype(int),
#     'VideoFilename': subj_df['#VideoFilename'].astype(str),
#     }

#     # Transform the dictionary created above into a new DataFrame
#     subj_df = pd.DataFrame(subj_df)

#     # Calculate all the seconds uploaded
#     subj_df["seconds"] = [list(range(i, i+j, 1)) for i, j in subj_df[['upl_seconds','clip_length']].values]

#     # Reshape the dataframe of potential seconds for the new clips to start
#     subj_df = expand_list(subj_df, "seconds", "upl_seconds").drop(columns=['clip_length'])


#     # Estimate the maximum number of clips available
#     survey_end_movie = movie_df["survey_end"].values[0]
#     max_n_clips = math.floor(survey_end_movie/clip_length)
    
#     ###### Select number of clips ##########
#     # Display the number of potential clips available
#     n_clips = widgets.IntSlider(
#         value=max_n_clips,
#         min=0,
#         max=max_n_clips,
#         step=1,
#         description = 'Number of clips to upload:',
#     )
    
#     display(n_clips)
    
#     return n_clips

# def choose_subjectset_method():
    
#     # Specify whether to upload to a new or existing workflow 
#     subjectset_method = widgets.ToggleButtons(
#         options=['Existing','New'],
#         description='Subjectset destination:',
#         disabled=False,
#         button_style='success',
#     )
    
#     display(subjectset_method)
#     return subjectset_method

# def choose_subjectset(df, method):
    
#     if method=="Existing":
#         # Select subjectset availables
#         subjectset = widgets.Combobox(
#             options=list(df.subject_set_id.apply(str).unique()),
#             description='Subjectset id:',
#             ensure_option=True,
#             disabled=False,
#         )
#     else:
#         # Specify the name of the new subjectset
#         subjectset = widgets.Text(
#             placeholder='Type subjectset name',
#             description='New subjectset name:',
#             disabled=False
#         )
        
#     display(subjectset)
#     return subjectset






# def choose_subject_set1(subjectset_df):
    
#     # Specify whether to upload to a new or existing workflow 
#     subjectset_method = widgets.ToggleButtons(
#         options=['Existing','New'],
#         description='Subjectset destination:',
#         disabled=False,
#         button_style='success',
#     )
#     output = widgets.Output()
    
#     def on_button_clicked(method):
#         with output:
#             if method['new']=="Existing":
#                 output.clear_output()
#                 # Select subjectset availables
#                 subjectset = widgets.Combobox(
#                     options=list(subjectset_df.subject_set_id.apply(str).unique()),
#                     description='Subjectset id:',
#                     ensure_option=True,
#                     disabled=False,
#                 )
                
#                 display(subjectset)

#                 return subjectset
                
#             else:
#                 output.clear_output()
#                 # Specify the name of the new subjectset
#                 subjectset = widgets.Text(
#                     placeholder='Type subjectset name',
#                     description='New subjectset name:',
#                     disabled=False
#                 )
            
            
#                 display(subjectset)

#                 return subjectset
            
            
#     subjectset_method.observe(on_button_clicked, names='value')
    
#     display(subjectset_method, output)
