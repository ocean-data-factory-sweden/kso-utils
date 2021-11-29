import os, cv2, sys, io
import operator
import pandas as pd
from tqdm import tqdm
import utils.server_utils as server_utils
import utils.spyfish_utils as spyfish_utils


# Calculate length and fps of a movie
def get_length(video_file):
    
    #final_fn = video_file if os.path.isfile(video_file) else koster_utils.unswedify(video_file)
    
    if os.path.isfile(video_file):
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)     
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        length = frame_count/fps
    else:
        print("Length and fps for", video_file, "were not calculated")
        length, fps = None, None
        
    return fps, length


def check_fps_duration(df, movies_csv, project_name):
    
    # Check if fps or duration is missing from any movie
    if not df[["fps", "duration"]].isna().all().any():
        
        print("Fps and duration information checked")
        
    else:

        # Select only those movies with the missing parameters
        miss_par_df = df[df[["fps", "duration"]].isna()]
        
        print("Updating the fps and duration information of:")
        print(*miss_par_df.filename.unique(), sep = "\n")
        
        ##### Check if movies with missing fps/duration info can be mapped ####
        # Add info about accessing the Spyfish movies from AWS
        if project_name == "Spyfish_Aotearoa":
            # Start AWS session
            aws_access_key_id, aws_secret_access_key = server_utils.aws_credentials()
            client = server_utils.connect_s3(aws_access_key_id, aws_secret_access_key)

            # Check the movies are accessible
            miss_par_df = spyfish_utils.check_spyfish_movies(miss_par_df, client)

        # Add info about accessing the Koster movies
        if project_name == "Koster_Seafloor_Obs":
            # Specify the path of the movies 
            movies_path = "/uploads"

            # Include server's path to the movie files
            miss_par_df["Fpath"] = movies_path + "/" + miss_par_df["filename"]

            # Check that videos can be mapped
            miss_par_df['exists'] = miss_par_df['Fpath'].map(os.path.isfile)

        # Prevent missing parameters from movies that don't exists
        if len(miss_par_df[~miss_par_df.exists]) > 0:
            print(
                f"There are {len(miss_par_df) - miss_par_df.exists.sum()} out of {len(miss_par_df)} movies missing from the server without {parameter} information. The movies are {miss_par_df[~miss_par_df.exists].filename.tolist()}"
            )

            return

        ##### Estimate the fps/duration of the movies ####
        else:
            # Check if the project is the Spyfish Aotearoa
            if project_name == "Spyfish_Aotearoa":
                # Download from s3, calculate and add fps/length info
                df = spyfish_utils.add_fps_length_spyfish(df, miss_par_df, client)

            else:    
                # Set the fps and duration of each movie
                df.loc[df["fps"].isna()|df["duration"].isna(), "fps": "duration"] = pd.DataFrame(df["Fpath"].apply(get_length, 1).tolist(), columns=["fps", "duration"])
            
        # Update the local movies.csv file with the new fps/duration info
        df.drop(["Fpath","exists"], axis=1).to_csv(movies_csv, index=False)
        print("The fps and duration columns have been updated in movies.csv")
        
    return df
                    
     
        
def check_survey_start_end(df, movies_csv):
    # Check if survey start or end is missing from any movie
    if not df[["survey_start", "survey_end"]].isna().all().any():
        
        print("Survey_start and survey_end information checked")
        
    else:
        
        print("Updating the survey_start or survey_end information of:")
        print(*df[df[["survey_start", "survey_end"]].isna()].filename.unique(), sep = "\n")
        
        # Set the start of each movie to 0 if empty
        df.loc[df["survey_start"].isna(),"survey_start"] = 0

            
        # Set the end of each movie to the duration of the movie if empty
        df.loc[df["survey_end"].isna(),"survey_end"] = df["duration"]

        # Update the local movies.csv file with the new survey start/end info
        df.to_csv(movies_csv, index=False)
        
        print("The survey start and end columns have been updated in movies.csv")

        
    # Prevent ending survey times longer than actual movies
    if (df["survey_end"] > df["duration"]).any():
        print("The survey_end times of the following movies are longer than the actual movies")
        print(*df[df["survey_end"] > df["duration"]].filename.unique(), sep = "\n")

    return df
                    
    
def get_movie_extensions():
    # Specify the formats of the movies to select
    movie_formats = tuple(['wmv', 'mpg', 'mov', 'avi', 'mp4', 'MOV', 'MP4'])
    
    return movie_formats


    
    