# base imports
import pandas as pd
import ipysheet
import datetime
import os
import subprocess

# widget imports
from IPython.display import display
from ipywidgets import interactive, Layout, HBox
import ipywidgets as widgets

# util imports
import kso_utils.db_utils as db_utils
import kso_utils.movie_utils as movie_utils
import kso_utils.spyfish_utils as spyfish_utils
import kso_utils.server_utils as server_utils


def open_sites_csv(db_initial_info):
    # Load the csv with sites information
    sites_df = pd.read_csv(db_initial_info["local_sites_csv"])

    # Load the df as ipysheet
    sheet = ipysheet.from_dataframe(sites_df)

    return sheet
    

def check_sites_database(db_initial_info, sites_df_sheet, project):

    # Load the csv with sites information
    sites_df = ipysheet.to_dataframe(sites_df_sheet)
    
    # Check if the project is the Spyfish Aotearoa
    if project.Project_name == "Spyfish_Aotearoa":
        # Rename columns to match schema fields
        sites_df = spyfish_utils.process_spyfish_sites(sites_df)
        
    # Select relevant fields
    sites_df = sites_df[
        ["site_id", "siteName", "decimalLatitude", "decimalLongitude", "geodeticDatum", "countryCode"]
    ]
    
    # Roadblock to prevent empty lat/long/datum/countrycode
    db_utils.test_table(
        sites_df, "sites", sites_df.columns
    )
    
    print("sites.csv file is all good!")

def open_movies_csv(db_initial_info):
    # Load the csv with movies information
    movies_df = pd.read_csv(db_initial_info["local_movies_csv"])

    # Load the df as ipysheet
    sheet = ipysheet.from_dataframe(movies_df)

    return sheet


def check_movies_csv(db_initial_info, movies_df_sheet, project):

    # Check for missing fps and duration info
    movies_df = movie_utils.check_fps_duration(db_initial_info, project)
    
    # Check if the project is the Spyfish Aotearoa
    if project.Project_name == "Spyfish_Aotearoa":
        movies_df = spyfish_utils.process_spyfish_movies(movies_df)
        
        
    # Check if the project is the KSO
    if project.Project_name == "Koster_Seafloor_Obs":
        movies_df = koster_utils.process_koster_movies_csv(movies_df)
    
    # Check if project is the template
    if project.Project_name == "Template project":
        # Add path of the movies
        movies_df["Fpath"] = "https://www.wildlife.ai/wp-content/uploads/2022/05/"+ movies_df["filename"]
    
    # Connect to database
    conn = db_utils.create_connection(db_initial_info['db_path'])
    
    # Reference movies with their respective sites
    sites_df = pd.read_sql_query("SELECT id, siteName FROM sites", conn)
    sites_df = sites_df.rename(columns={"id": "Site_id"})

    # Merge movies and sites dfs
    movies_df = pd.merge(
        movies_df, sites_df, how="left", on="siteName"
    )
    
    # Select only those fields of interest
    movies_db = movies_df[
        ["movie_id", "filename", "created_on", "fps", "duration", "sampling_start", "sampling_end", "Author", "Site_id", "Fpath"]
    ]

    # Roadblock to prevent empty information
    db_utils.test_table(
        movies_db, "movies", movies_db.columns
    )
    
    # Check for sampling_start and sampling_end info
    movies_df = movie_utils.check_sampling_start_end(movies_df, db_initial_info)
    
    # Ensure date is ISO 8601:2004(E) and compatible with Darwin Data standards
    date_time_check = pd.to_datetime(movies_df.created_on, infer_datetime_format=True)
#     print("The last dates from the created_on column are:")
#     print(date_time_check.tail())
   
    print("movies.csv is all good!") 
    


def check_movies_from_server(db_info_dict, project):
    # Load the csv with movies information
    movies_df = pd.read_csv(db_info_dict["local_movies_csv"]) 
    
    # Check if the project is the Spyfish Aotearoa
    if project.Project_name == "Spyfish_Aotearoa":
        # Retrieve movies that are missing info in the movies.csv
        missing_info = spyfish_utils.check_spyfish_movies(movies_df, db_info_dict)
        
#     print(missing_info)
    # Find out files missing from the Server
    missing_from_server = missing_info[missing_info["_merge"]=="left_only"]
#     missing_bad_deployment = missing_from_server[missing_from_server["IsBadDeployment"]]
#     missing_no_bucket_info = missing_from_server[~(missing_from_server["IsBadDeployment"])]
    
    print("There are", len(missing_from_server.index), "movies missing")
#     print(len(missing_bad_deployment.index), "movies are bad deployments. Their filenames are:")
#     print(*missing_bad_deployment.filename.unique(), sep = "\n")
#     print(len(missing_no_bucket_info.index), "movies are good deployments but don't have movies uploaded. Their filenames are:")
#     print(*missing_no_bucket_info.filename.unique(), sep = "\n")
    
    # Find out files missing from the csv
    missing_from_csv = missing_info[missing_info["_merge"]=="right_only"].reset_index(drop=True)
            
    print("There are", len(missing_from_csv.index), "movies missing from movies.csv. Their filenames are:")
#     print(*missing_from_csv.filename.unique(), sep = "\n")
    
    return missing_from_server, missing_from_csv

def select_deployment(missing_from_csv):
    if missing_from_csv.shape[0]>0:        
        # Widget to select the deployment of interest
        deployment_widget = widgets.SelectMultiple(
            options = missing_from_csv.deployment_folder.unique(),
            description = 'New deployment:',
            disabled = False,
            layout=Layout(width='50%'),
            style = {'description_width': 'initial'}
        )
        display(deployment_widget)
        
        return deployment_widget
    
def select_eventdate():
    # Select the date 
    date_widget = widgets.DatePicker(
        description='Date of deployment:',
        value = datetime.date.today(),
        disabled=False,
        layout=Layout(width='50%'),
        style = {'description_width': 'initial'}
    )
    display(date_widget)
    
    return date_widget


def update_new_deployments(deployment_selected, db_info_dict, event_date):
    for deployment_i in deployment_selected.value:      
        print(f"Starting to concatenate {deployment_i} out of {len(deployment_selected.value)} deployments selected")
        
        # Get a dataframe of movies from the deployment
        movies_s3_pd = server_utils.get_matching_s3_keys(db_info_dict["client"], 
                                                     db_info_dict["bucket"], 
                                                     prefix = deployment_i,
                                                     suffix = movie_utils.get_movie_extensions())
        
        # Create a list of the list of movies inside the deployment selected
        movie_files_server = movies_s3_pd.Key.unique().tolist()
        
        
        if len(movie_files_server)<2:
            print(f"Deployment {deployment_i} will not be concatenated because it only has {movies_s3_pd.Key.unique()}")
        else:
            # Concatenate the files if multiple
            print("The files", movie_files_server, "will be concatenated")

            # Start text file and list to keep track of the videos to concatenate
            textfile_name = "a_file.txt"
            textfile = open(textfile_name, "w")
            video_list = []           

            for movie_i in sorted(movie_files_server):
                # Specify the temporary output of the go pro file
                movie_i_output = movie_i.split("/")[-1]

                # Download the files from the S3 bucket
                if not os.path.exists(movie_i_output):
                    server_utils.download_object_from_s3(client = db_info_dict["client"],
                                                         bucket = db_info_dict["bucket"], 
                                                         key=movie_i,
                                                         filename=movie_i_output,
                                                        )
                # Keep track of the videos to concatenate 
                textfile.write("file '"+ movie_i_output + "'"+ "\n")
                video_list.append(movie_i_output)
            textfile.close()

      
            # Save eventdate as str
            EventDate_str = event_date.value.strftime("%d_%m_%Y")

            # Specify the name of the concatenated video
            filename = deployment_i.split("/")[-1]+"_"+EventDate_str+".MP4"

            # Concatenate the files
            if not os.path.exists(filename):
                print("Concatenating ",filename)

                # Concatenate the videos
                subprocess.call(["ffmpeg", 
                                 "-f", "concat", 
                                 "-safe", "0",
                                 "-i", "a_file.txt", 
                                 "-c", "copy", 
                                 filename])
                
            # Upload the concatenated video to the S3
            server_utils.upload_file_to_s3(
                db_info_dict["client"],
                bucket=db_info_dict["bucket"],
                key=deployment_i+"/"+filename,
                filename=filename,
            )

            print(filename, "successfully uploaded to", deployment_i)

            # Delete the raw videos downloaded from the S3 bucket
            for f in video_list:
                os.remove(f)

            # Delete the text file
            os.remove(textfile_name)

            # Delete the concat video
            os.remove(concat_video)


        
        
# def upload_movies():
    
#     # Define widget to upload the files
#     mov_to_upload = widgets.FileUpload(
#         accept='.mpg',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
#         multiple=True  # True to accept multiple files upload else False
#     )
    
#     # Display the widget?
#     display(mov_to_upload)
    
#     main_out = widgets.Output()
#     display(main_out)
    
#     # TODO Copy the movie files to the movies folder
    
#     # Provide the site, location, date info of the movies
#     upload_info_movies()
#     print("uploaded")
    
# # Check that videos can be mapped
#     movies_df['exists'] = movies_df['Fpath'].map(os.path.isfile)    
    
# def upload_info_movies():

#     # Select the way to upload the info about the movies
#     widgets.ToggleButton(
#     value=False,
#     description=['I have a csv file with information about the movies',
#                  'I am happy to write here all the information about the movies'],
#     disabled=False,
#     button_style='success', # 'success', 'info', 'warning', 'danger' or ''
#     tooltip='Description',
#     icon='check'
# )
    
#     # Upload the information using a csv file
#     widgets.FileUpload(
#     accept='',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
#     multiple=False  # True to accept multiple files upload else False
# )
#     # Upload the information 
    
#     # the folder where the movies are
    
#     # Try to extract location and date from the movies 
#     widgets.DatePicker(
#     description='Pick a Date',
#     disabled=False
# )
    
#     # Run an interactive way to write metadata info about the movies
    
#     print("Thanks for providing all the required information about the movies")
    
    
    
    


# # Select multiple movies to include information of
# def go_pro_movies_to_update(df):
    
#     # Save the filenames of the movies missing
#     filename_missing_csv = df.location_and_filename.unique()
    
#     # Display the project options
#     movie_to_update = widgets.SelectMultiple(
#         options=filename_missing_csv,
#         rows=15,
#         layout=Layout(width='80%'),
#         description="GO pro movies:",
#         disabled=False,
        
#     )
    
#     display(movie_to_update)
#     return movie_to_update

# # Select one movie to include information of
# def full_movie_to_update(df):
    
#     # Save the filenames of the movies missing
#     filename_missing_csv = df.location_and_filename.unique()
    
#     # Display the project options
#     movie_to_update = widgets.Dropdown(
#         options=filename_missing_csv,
#         rows=15,
#         layout=Layout(width='80%'),
#         description="Full movie:",
#         disabled=False,
        
#     )
    
#     display(movie_to_update)
#     return movie_to_update


# # Select the info to add to the csv
# def info_to_csv(df, movies):
    
#     # Save the filenames of the movies missing
#     filename_missing_csv = df.location_and_filename.unique()
    
#     # Display the project options
#     movie_to_update = widgets.SelectMultiple(
#         options=filename_missing_csv,
#         rows=15,
#         layout=Layout(width='80%'),
#         description="Movie:",
#         disabled=False,
        
#     )
    
#     display(movie_to_update)
#     return movie_to_update
    
    