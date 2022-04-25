# base imports
import os, cv2
import pandas as pd
from tqdm import tqdm
import difflib

# util imports
import kso_utils.server_utils as server_utils
import kso_utils.db_utils as db_utils


# Calculate length and fps of a movie
def get_length(video_file, movie_folder):
    
    files = os.listdir(movie_folder)
    
    if os.path.basename(video_file) in files:
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)     
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        length = frame_count/fps
    else:
        print("Length and fps for", video_file, "were not calculated - probably missing")
        length, fps = None, None
        
    return fps, length


def check_fps_duration(db_info_dict, project):
    
    movie_folder = project.movie_folder
    
    # Load the csv with movies information
    df = pd.read_csv(db_info_dict["local_movies_csv"])
    
    # Check if fps or duration is missing from any movie
    if not df[["fps", "duration"]].isna().any().all():
        
        print("Fps and duration information checked")
        
    else:

        # Get project server
        server = project.server
        
        # Select only those movies with the missing parameters
        miss_par_df = df[df["fps"].isna()|df["duration"].isna()]
        
        print("Updating the fps and duration information of:")
        print(*miss_par_df.filename.unique(), sep = "\n")
        
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
            
            
            # Update the local and server movies.csv file with the new fps/duration info
            df.to_csv(db_info_dict["local_movies_csv"], index=False)
            server_utils.upload_file_to_s3(client = db_info_dict['client'],
                                           bucket = db_info_dict['bucket'], 
                                           key = db_info_dict["server_movies_csv"],
                                           filename = db_info_dict["local_movies_csv"])
            print("The fps and duration columns have been updated in movies.csv")

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
            
        print("Fps and duration information updated")
        
    return df

     

def check_sampling_start_end(df, db_info_dict):
    # Load the csv with movies information
    movies_csv = pd.read_csv(db_info_dict["local_movies_csv"])
    
    # Check if sampling start or end is missing from any movie
    if not df[["sampling_start", "sampling_end"]].isna().all().any():
        
        print("sampling_start and survey_end information checked")
        
    else:
        
        print("Updating the survey_start or survey_end information of:")
        print(*df[df[["sampling_start", "sampling_end"]].isna()].filename.unique(), sep = "\n")
        
        # Set the start of each movie to 0 if empty
        df.loc[df["sampling_start"].isna(),"sampling_start"] = 0

            
        # Set the end of each movie to the duration of the movie if empty
        df.loc[df["survey_end"].isna(),"sampling_end"] = df["duration"]

        # Update the local movies.csv file with the new sampling start/end info
        df.to_csv(movies_csv, index=False)
        
        print("The survey start and end columns have been updated in movies.csv")

        
    # Prevent ending sampling times longer than actual movies
    if (df["sampling_end"] > df["duration"]).any():
        print("The sampling_end times of the following movies are longer than the actual movies")
        print(*df[df["sampling_end"] > df["duration"]].filename.unique(), sep = "\n")

    return df


def get_movie_extensions():
    # Specify the formats of the movies to select
    movie_formats = tuple(['wmv', 'mpg', 'mov', 'avi', 'mp4', 'MOV', 'MP4'])
    
    return movie_formats



    
