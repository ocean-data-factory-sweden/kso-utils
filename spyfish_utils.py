#spyfish utils
import os
import sqlite3
import pandas as pd
import numpy as np

import kso_utils.server_utils as server_utils
import kso_utils.movie_utils as movie_utils
import kso_utils.db_utils as db_utils
from tqdm import tqdm
import subprocess
from pathlib import Path

def check_spyfish_movies(movies_df, client, bucket_i):
    
    # Get dataframe of movies from AWS
    movies_s3_pd = server_utils.get_matching_s3_keys(client, bucket_i, suffix=movie_utils.get_movie_extensions())

    # Specify the key of the movies (path in S3 of the object)
    movies_df["Key"] = movies_df["prefix"] + filename

    # Missing info for files in the "buv-zooniverse-uploads"
    movies_df = movies_df.merge(movies_s3_pd["Key"], 
                                on=['Key'], how='left', 
                                indicator=True)

    # Check that movies can be mapped
    movies_df['exists'] = np.where(movies_df["_merge"]=="left_only", False, True)
        
    # Drop _merge columns to match sql squema
    movies_df = movies_df.drop("_merge", axis=1)
        
    return movies_df
                       
def add_fps_length_spyfish(df, miss_par_df, client):
    
    # Loop through each movie missing fps and duration
    for index, row in tqdm(miss_par_df.iterrows(), total=miss_par_df.shape[0]):
        if not os.path.exists(row['filename']):
            # Download the movie locally
            server_utils.download_object_from_s3(
                client,
                bucket='marine-buv',
                key=row['Key'],
                filename=row['filename'],
            )

        # Set the fps and duration of the movie
        df.at[index,"fps"], df.at[index, "duration"] = movie_utils.get_length(row['filename'])
        
        # Delete the downloaded movie
        os.remove(row['filename'])
                    
                    
    return df
    
def process_spyfish_sites(sites_df):
    
    # Rename relevant fields
    sites_df = sites_df.rename(
        columns = {
            "schema_site_id": "site_id",# site id for the db
            "SiteID": "siteName",#site id used for zoo subjects
            "Latitude": "decimalLatitude",
            "Longitude": "decimalLongitude"
        }
    )
        
    return sites_df


def process_spyfish_movies(movies_df):
    
    # Rename relevant fields
    movies_df = movies_df.rename(
        columns = {
            "LinkToVideoFile": "Fpath",
            "EventDate": "created_on",
            "SamplingStart": "sampling_start",
            "SamplingEnd": "sampling_end",
            "RecordedBy": "Author",
            "SiteID": "siteName",

        }
    )

    # Remove extension from the filename to match the subject metadata from Zoo
    movies_df["filename"] = movies_df["filename"].str.split('.',1).str[0]
    
    
    return movies_df
    
    

# Function to download go pro videos, concatenate them and upload the concatenated videos to aws 
def concatenate_videos(df, session):

    # Loop through each survey to find out the raw videos recorded with the GoPros
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        
        # Select the go pro videos from the "i" survey to concatenate
        list1 = row['go_pro_files'].split(';')
        list_go_pro = [row['prefix'] + "/" + s for s in list1]

        # Start text file and list to keep track of the videos to concatenate
        textfile_name = "a_file.txt"
        textfile = open(textfile_name, "w")
        video_list = []

        print("Downloading", len(list_go_pro), "videos")

        # Download each go pro video from the S3 bucket
        for go_pro_i in tqdm(list_go_pro, total=len(list_go_pro)):
            
            # Specify the temporary output of the go pro file
            go_pro_output = go_pro_i.split("/")[-1]

            # Download the files from the S3 bucket
            if not os.path.exists(go_pro_output):
                server_utils.download_object_from_s3(
                    session,
                    bucket=row['bucket'],
                    key=go_pro_i,
                    filename=go_pro_output,
                )
                
                #client.download_file(bucket_i, go_pro_i, go_pro_output)

            # Keep track of the videos to concatenate 
            textfile.write("file '"+ go_pro_output + "'"+ "\n")
            video_list.append(go_pro_output)

        textfile.close()

        concat_video = row['filename']

        if not os.path.exists(concat_video):

            print("Concatenating ",concat_video)

            # Concatenate the videos
            subprocess.call(["ffmpeg", 
                             "-f", "concat", 
                             "-safe", "0",
                             "-i", "a_file.txt", 
                             "-c", "copy", 
                             #"-an",#removes the audio
                             concat_video])
            
        print(concat_video, "concatenated successfully")

        # Upload the concatenated video to the S3
        s3_destination = row['prefix'] + "/" + concat_video
        server_utils.upload_file_to_s3(
            session,
            bucket=row['bucket'],
            key=s3_destination,
            filename=concat_video,
        )
                    
        print(concat_video, "succesfully uploaded to", s3_destination)
        
        # Delete the raw videos downloaded from the S3 bucket
        for f in video_list:
            os.remove(f)

        # Delete the text file
        os.remove(textfile_name)
        
        # Update the fps and length info
        #movie_utils.get_length(concat_video)
        
        # Delete the concat video
        os.remove(concat_video)

        print("Temporary files and videos removed")
       
    
def process_spyfish_subjects(subjects, db_path):
    
    # Merge "#Subject_type" and "Subject_type" columns to "subject_type"
    subjects['subject_type'] = subjects['Subject_type'].fillna(subjects['#Subject_type'])
    
    # Rename columns to match the db format
    subjects = subjects.rename(
        columns={
            "#VideoFilename": "filename",
            "upl_seconds": "clip_start_time",
            "#frame_number": "frame_number"
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
        columns={
            "id": "frame_exp_sp_id",
            "scientificName": "ScientificName"
        }
    )
    
    # Reference the expected species on the uploaded subjects
    subjects = pd.merge(subjects.drop(columns=['frame_exp_sp_id']), species_df, how="left", on="ScientificName")

    ##### Match site code to name from movies sql and get movie_id to save it as "movie_id"
    # Query id and filenames from the movies table
    movies_df = pd.read_sql_query("SELECT id, filename FROM movies", conn)
    
    # Rename columns to match subject df 
    movies_df = movies_df.rename(
        columns={
            "id": "movie_id"
        }
    )
    
    # Reference the movienames with the id movies table
    subjects = pd.merge(subjects, movies_df, how="left", on="filename")
    
    return subjects

        
def process_clips_spyfish(annotations, row_class_id, rows_list):
    
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



def get_spyfish_choices(server_dict, db_initial_info, db_csv_info):
    # Get the server path of the csv with sites and survey choices
    server_choices_csv = server_utils.get_matching_s3_keys(server_dict["client"],
                                              db_initial_info["bucket"],
                                              prefix = db_initial_info["key"]+"/"+"choices")['Key'][0]


    # Specify the local path for the csv
    local_choices_csv = str(Path(db_csv_info,Path(server_choices_csv).name))

    # Download the csv
    server_utils.download_object_from_s3(server_dict["client"],
                            bucket=db_initial_info["bucket"],
                            key=server_choices_csv, 
                            filename=local_choices_csv)

    db_initial_info["server_choices_csv"] = server_choices_csv
    db_initial_info["local_choices_csv"] = Path(local_choices_csv)
    
    return db_initial_info


def spyfish_subject_metadata(df, db_info_dict):
    
    # Get extra movie information
    movies_df = pd.read_csv(db_info_dict["local_movies_csv"])
    
    df = df.merge(movies_df.drop(columns=["filename"]), how="left", on="movie_id")
    
    # Get extra survey information
    surveys_df = pd.read_csv(db_info_dict["local_surveys_csv"])
    
    df = df.merge(surveys_df, how="left", on="SurveyID")
    
    # Get extra site information
    sites_df = pd.read_csv(db_info_dict["local_sites_csv"])
    
    df = df.merge(sites_df.drop(columns=["LinkToMarineReserve"]), how="left", on="SiteID")
        
    # Convert datetime to string to avoid JSON seriazible issues
    df['EventDate'] = df['EventDate'].astype(str)

    df = df.rename(columns={
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
        })

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
    

