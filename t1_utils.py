# base imports
import os
import subprocess
import pandas as pd
import numpy as np
import datetime
import logging
from tqdm import tqdm

# widget imports
from IPython.display import display
from ipywidgets import interactive, Layout, HBox
import ipywidgets as widgets
import ipysheet
import folium
from folium.plugins import MiniMap
import asyncio

# util imports
import kso_utils.db_utils as db_utils
import kso_utils.movie_utils as movie_utils
import kso_utils.spyfish_utils as spyfish_utils
import kso_utils.server_utils as server_utils
import kso_utils.tutorials_utils as t_utils
import kso_utils.koster_utils as koster_utils
import kso_utils.project_utils as project_utils

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

out_df = pd.DataFrame()


####################################################    
############ CSV/iPysheet FUNCTIONS ################
####################################################

def select_sheet_range(db_info_dict: dict, orig_csv: str):
    """
    > This function loads the csv file of interest into a pandas dataframe and enables users to pick a range of (rows) to display
    
    :param db_info_dict: a dictionary with the following keys:
    :param orig_csv: the original csv file name
    :type orig_csv: str
    :return: A dataframe with the sites information
    """
    
    # Load the csv with the information of interest
    df = pd.read_csv(db_info_dict[orig_csv])
    
    df_range = widgets.SelectionRangeSlider(
                          options=range(0, len(df.index)+1),
                          index=(0,len(df. index)),
                          description='Rows to display',
                          orientation='horizontal',
                          layout=Layout(width='90%', padding='35px'),
                          style = {'description_width': 'initial'}
                          )

    display(df_range)

    return df, df_range                   

def open_csv(df: pd.DataFrame, df_range: widgets.Widget):
    """
    > This function loads the dataframe with the information of interest, filters the range of rows selected and then loads the dataframe into
    an ipysheet
    
    :param df: a pandas dataframe of the information of interest:
    :param df_range: the range widget selection :
    :return: A (subset) dataframe with the information of interest and the same data in an interactive sheet
    """
    # Extract the first and last row to display
    range_start = int(df_range.label[0])
    range_end = int(df_range.label[1])

    # Display the range of sites selected
    logging.info(f"Displaying # {range_start} to # {range_end}")
    
    # Filter the dataframe based on the selection
    df_filtered = df.filter(items = range(range_start, range_end), axis=0)

    # Load the df as ipysheet
    sheet = ipysheet.from_dataframe(df_filtered)

    return df_filtered, sheet
    
    
def display_changes(db_info_dict: dict, isheet: ipysheet.Sheet, df_filtered: pd.DataFrame):
    """
    It takes the dataframe from the ipysheet and compares it to the dataframe from the local csv file.
    If there are any differences, it highlights them and returns the dataframe with the changes
    highlighted
    
    :param db_info_dict: a dictionary containing the database information
    :type db_info_dict: dict
    :param isheet: The ipysheet object that contains the data
    :param sites_df_filtered: a pandas dataframe with information of a range of sites
    :return: A tuple with the highlighted changes and the sheet_df
    """
    # Convert ipysheet to pandas
    sheet_df = ipysheet.to_dataframe(isheet)
    
    # Check the differences between the modified and original spreadsheets
    sheet_diff_df = pd.concat([df_filtered, sheet_df]).drop_duplicates(keep=False)
    
    # If changes in dataframes display them and ask the user to confirm them
    if sheet_diff_df.empty:
        logging.error("There are no changes to update")
        raise
    else:
        # Retieve the column name of the id of interest (Sites, movies,..)
        id_col = [col for col in df_filtered.columns if '_id' in col][0]
        
        # Concatenate DataFrames and distinguish each frame with the keys parameter
        df_all = pd.concat([df_filtered.set_index(id_col), sheet_df.set_index(id_col)],
            axis='columns', keys=['Origin', 'Update'])
        
        # Rearrange columns to have them next to each other
        df_final = df_all.swaplevel(axis='columns')[df_filtered.columns[1:]]
        

        # Create a function to highlight the changes
        def highlight_diff(data, color='yellow'):
            attr = 'background-color: {}'.format(color)
            other = data.xs('Origin', axis='columns', level=-1)
            return pd.DataFrame(np.where(data.ne(other, level=0), attr, ''),
                                index=data.index, columns=data.columns)

        # Return the df with the changes highlighted
        highlight_changes = df_final.style.apply(highlight_diff, axis=None)

        return highlight_changes, sheet_df

def update_csv(db_info_dict: dict, project: project_utils.Project, sheet_df: pd.DataFrame, df: pd.DataFrame, local_csv: str, serv_csv: str):
    """
    This function is used to update the csv files locally and in the server
    
    :param db_info_dict: The dictionary containing the database information
    :param project: The project object
    :param sheet_df: The dataframe of the sheet you want to update
    :param df: a pandas dataframe of the information of interest
    :param local_csv: a string of the names of the local csv to update
    :param serv_csv: a string of the names of the server csv to update
    """
    # Create button to confirm changes
    confirm_button = widgets.Button(
      description = 'Yes, details are correct',
      layout=Layout(width='25%'),
      style = {'description_width': 'initial'},
      button_style='danger'
      )

    # Create button to deny changes
    deny_button = widgets.Button(
        description = 'No, I will go back and fix them',
        layout=Layout(width='45%'),
        style = {'description_width': 'initial'}, 
        button_style='danger'
    )

    # Save changes in survey csv locally and in the server
    async def f(sheet_df, df, local_csv, serv_csv):
        x = await t_utils.wait_for_change(confirm_button,deny_button) #<---- Pass both buttons into the function
        if x == "Yes, details are correct": #<--- use if statement to trigger different events for the two buttons
            logging.info("Checking if changes can be incorporated to the database")
            
            # Retieve the column name of the id of interest (Sites, movies,..)
            id_col = [col for col in df.columns if '_id' in col][0]
            
            # Replace the different values based on id
            df.set_index(id_col, inplace=True)
            sheet_df.set_index(id_col, inplace=True)
            df.update(sheet_df)
            df.reset_index(drop=False, inplace=True)
            
            # Process the csv of interest and tests for compatibility with sql table 
            csv_i, df_to_db = db_utils.process_test_csv(db_info_dict= db_info_dict, 
                                                        project = project,
                                                        local_csv = local_csv)
            
            # Save the updated df locally
            df.to_csv(db_info_dict[local_csv], index=False)
            logging.info("The local csv file has been updated")
            
            # Save the updated df in the server
            server_utils.update_csv_server(project, db_info_dict, orig_csv = serv_csv, updated_csv = local_csv)
                        
        else:
            logging.info("Run this cell again when the changes are correct!")

    print("")
    print("Are the changes above correct?")
    display(HBox([confirm_button,deny_button])) #<----Display both buttons in an HBox
    asyncio.create_task(f(sheet_df, df, local_csv, serv_csv))


####################################################    
############### SITES FUNCTIONS ###################
####################################################
def map_site(db_info_dict: dict, project: project_utils.Project):
    """
    > This function takes a dictionary of database information and a project object as input, and
    returns a map of the sites in the database
    
    :param db_info_dict: a dictionary containing the information needed to connect to the database
    :type db_info_dict: dict
    :param project: The project object
    :return: A map with all the sites plotted on it.
    """
    if project.server == "SNIC":
      # Set initial location to Gothenburg 
      init_location = [57.708870, 11.974560]
    
    else:
      # Set initial location to Taranaki
      init_location = [-39.296109, 174.063916]

    # Create the initial kso map
    kso_map = folium.Map(location=init_location,width=900,height=600)

    # Read the csv file with site information
    sites_df = pd.read_csv(db_info_dict["local_sites_csv"])

    # Combine information of interest into a list to display for each site
    sites_df["site_info"] = sites_df.values.tolist()

    # Save the names of the columns
    df_cols = sites_df.columns

    # Add each site to the map 
    sites_df.apply(lambda row:folium.CircleMarker(location=[row[df_cols.str.contains("Latitude")], 
                                                            row[df_cols.str.contains("Longitude")]], 
                                                            radius = 14, popup=row["site_info"], 
                                                            tooltip=row[df_cols.str.contains("siteName", 
                                                            case=False)]).add_to(kso_map), axis=1)

    # Add a minimap to the corner for reference
    kso_map = kso_map.add_child(MiniMap())
    
    # Return the map
    return kso_map

####################################################    
############### MOVIES FUNCTIONS ###################
####################################################

def choose_movie_review():
    """
    This function creates a widget that allows the user to choose between two methods to review the
    movies.csv file.
    :return: The widget is being returned.
    """
    choose_movie_review_widget = widgets.RadioButtons(
          options=["Basic: Automatic check for empty fps/duration and sampling start/end cells in the movies.csv","Advanced: Basic + Check format and metadata of each movie"],
          description='What method you want to use to review the movies:',
          disabled=False,
          layout=Layout(width='95%'),
          style = {'description_width': 'initial'}
      )
    display(choose_movie_review_widget)
    
    return choose_movie_review_widget

def check_movies_csv(db_info_dict: dict, project: project_utils.Project, review_method: widgets.Widget, gpu_available: bool = False):
    """
    > The function `check_movies_csv` loads the csv with movies information and checks if it is empty
    
    :param db_info_dict: a dictionary with the following keys:
    :param project: The project name
    :param review_method: The method used to review the movies
    :param gpu_available: Boolean, whether or not a GPU is available
    """
    
    # Load the csv with movies information
    df = pd.read_csv(db_info_dict["local_movies_csv"])

    # Check if the project is the Spyfish Aotearoa
    if project.Project_name == "Spyfish_Aotearoa":
        # Rename columns to match squema requirements
        df = spyfish_utils.process_spyfish_movies(df)
        
    if review_method.value.startswith("Basic"):
        # Check if fps or duration is missing from any movie
        if not df[["fps", "duration", "sampling_start", "sampling_end"]].isna().any().any():
            raise ValueError("Fps, duration and sampling information is not empty")
          
        else:
            # Create a df with only those rows with missing fps/duration
            df_missing = df[df["fps"].isna()|df["duration"].isna()].reset_index(drop=True)
            
            logging.info("Retrieving the paths to access the movies")
            # Add a column with the path (or url) where the movies can be accessed from
            df_missing["movie_path"] = pd.Series([movie_utils.get_movie_path(i, db_info_dict, project) for i in tqdm(df_missing["fpath"], total=df_missing.shape[0])])

            logging.info("Getting the fps and duration of the movies")
            # Read the movies and overwrite the existing fps and duration info 
            df_missing[["fps","duration"]] = pd.DataFrame([movie_utils.get_fps_duration(i) for i in tqdm(df_missing["movie_path"], total=df_missing.shape[0])], columns=["fps", "duration"])

            # Add the missing info to the original df based on movie ids
            df.set_index("movie_id", inplace=True)
            df_missing.set_index("movie_id", inplace=True)
            df.update(df_missing)
            df.reset_index(drop=False, inplace=True)
            
    else:
        logging.info("Retrieving the paths to access the movies")
        # Add a column with the path (or url) where the movies can be accessed from
        df["movie_path"] = pd.Series([movie_utils.get_movie_path(i, db_info_dict, project) for i in tqdm(df["fpath"], total=df.shape[0])])

        logging.info("Getting the fps and duration of the movies")
        # Read the movies and overwrite the existing fps and duration info 
        df[["fps","duration"]] = pd.DataFrame([movie_utils.get_fps_duration(i) for i in tqdm(df["movie_path"], total=df.shape[0])], columns=["fps", "duration"])

        logging.info("Standardising the format, frame rate and codec of the movies")

        # Convert movies to the right format, frame rate or codec and upload them to the project's server/storage
        [movie_utils.standarise_movie_format(i, j, k, db_info_dict, project, gpu_available) for i,j,k in tqdm(zip(df["movie_path"],df["filename"], df["fpath"]), total=df.shape[0])]
        
        # Drop unnecessary columns
        df = df.drop(columns=['movie_path'])
    
    # Fill out missing sampling start information
    df.loc[df["sampling_start"].isna(), "sampling_start"] = 0.0
    
    # Fill out missing sampling end information
    df.loc[df["sampling_end"].isna(), "sampling_end"] = df["duration"]

    # Prevent sampling end times longer than actual movies
    if (df["sampling_end"] > df["duration"]).any():
        mov_list = df[df["sampling_end"] > df["duration"]].filename.unique()
        raise ValueError(f"The sampling_end times of the following movies are longer than the actual movies {mov_list}")
    
    # Save the updated df locally
    df.to_csv(db_info_dict["local_movies_csv"], index=False)
    logging.info("The local movies.csv file has been updated")
    
    # Save the updated df in the server
    server_utils.update_csv_server(project, db_info_dict, orig_csv = "server_movies_csv", updated_csv = "local_movies_csv")
    
def check_movies_from_server(db_info_dict: dict, project: project_utils.Project):
    """
    It takes in a dataframe of movies and a dictionary of database information, and returns two
    dataframes: one of movies missing from the server, and one of movies missing from the csv
    
    :param db_info_dict: a dictionary with the following keys:
    :param project: the project object
    """
    # Load the csv with movies information
    movies_df = pd.read_csv(db_info_dict["local_movies_csv"]) 
    
    # Check if the project is the Spyfish Aotearoa
    if project.Project_name == "Spyfish_Aotearoa":
        # Retrieve movies that are missing info in the movies.csv
        missing_info = spyfish_utils.check_spyfish_movies(movies_df, db_info_dict)
        
    # Find out files missing from the Server
    missing_from_server = missing_info[missing_info["_merge"]=="left_only"]
    
    logging.info(f"There are {len(missing_from_server.index)} movies missing")
    
    # Find out files missing from the csv
    missing_from_csv = missing_info[missing_info["_merge"]=="right_only"].reset_index(drop=True)
            
    logging.info(f"There are {len(missing_from_csv.index)} movies missing from movies.csv. Their filenames are:{missing_from_csv.filename.unique()}")
    
    return missing_from_server, missing_from_csv

def select_deployment(missing_from_csv: pd.DataFrame):
    """
    > This function takes a dataframe of missing files and returns a widget that allows the user to
    select the deployment of interest
    
    :param missing_from_csv: a dataframe of the files that are missing from the csv file
    :return: A widget object
    """
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
    """
    > This function creates a date picker widget that allows the user to select a date. 
    
    The function is called `select_eventdate()` and it returns a date picker widget. 
    """
    date_widget = widgets.DatePicker(
        description='Date of deployment:',
        value = datetime.date.today(),
        disabled=False,
        layout=Layout(width='50%'),
        style = {'description_width': 'initial'}
    )
    display(date_widget)
    return date_widget


def update_new_deployments(deployment_selected: widgets.Widget, db_info_dict: dict, event_date: widgets.Widget):
    """
    It takes a deployment, downloads all the movies from that deployment, concatenates them, uploads the
    concatenated video to the S3 bucket, and deletes the raw movies from the S3 bucket
    
    :param deployment_selected: the deployment you want to concatenate
    :param db_info_dict: a dictionary with the following keys:
    :param event_date: the date of the event you want to concatenate
    """
    for deployment_i in deployment_selected.value:      
        logging.info(f"Starting to concatenate {deployment_i} out of {len(deployment_selected.value)} deployments selected")
        
        # Get a dataframe of movies from the deployment
        movies_s3_pd = server_utils.get_matching_s3_keys(db_info_dict["client"], 
                                                     db_info_dict["bucket"], 
                                                     prefix = deployment_i,
                                                     suffix = movie_utils.get_movie_extensions())
        
        # Create a list of the list of movies inside the deployment selected
        movie_files_server = movies_s3_pd.Key.unique().tolist()
        
        
        if len(movie_files_server)<2:
            logging.info(f"Deployment {deployment_i} will not be concatenated because it only has {movies_s3_pd.Key.unique()}")
        else:
            # Concatenate the files if multiple
            logging.info(f"The files {movie_files_server} will be concatenated")

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
                logging.info("Concatenating ", filename)

                # Concatenate the videos
                subprocess.call(["ffmpeg", 
                                 "-f", "concat", 
                                 "-safe", "0",
                                 "-i", "a_file.txt", 
                                 "-c:a", "copy",
                                 "-c:v", "h264",
                                 "-crf", "22",
                                 filename])
                
            # Upload the concatenated video to the S3
            server_utils.upload_file_to_s3(
                db_info_dict["client"],
                bucket=db_info_dict["bucket"],
                key=deployment_i+"/"+filename,
                filename=filename,
            )

            logging.info(f"{filename} successfully uploaded to {deployment_i}")

            # Delete the raw videos downloaded from the S3 bucket
            for f in video_list:
                os.remove(f)

            # Delete the text file
            os.remove(textfile_name)

            # Delete the concat video
            os.remove(filename)
            
            # Delete the movies from the S3 bucket
            for movie_i in sorted(movie_files_server):
                server_utils.delete_file_from_s3(client = db_info_dict["client"],
                                    bucket = db_info_dict["bucket"], 
                                    key=movie_i,
                                   )

#############
#####Species#####
#################
def check_species_csv(db_info_dict: dict, project: project_utils.Project):
    """
    > The function `check_species_csv` loads the csv with species information and checks if it is empty
    
    :param db_info_dict: a dictionary with the following keys:
    :param project: The project name
    """
    # Load the csv with movies information
    species_df = pd.read_csv(db_info_dict["local_species_csv"])

    # Retrieve the names of the basic columns in the sql db
    conn = db_utils.create_connection(db_info_dict["db_path"])
    data = conn.execute(f"SELECT * FROM species")
    field_names = [i[0] for i in data.description]

    # Select the basic fields for the db check
    df_to_db = species_df[
        [c for c in species_df.columns if c in field_names]
    ]

    # Roadblock to prevent empty lat/long/datum/countrycode
    db_utils.test_table(
        df_to_db, "species", df_to_db.columns
    )

    logging.info("The species dataframe is complete")
#           
        
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
    
    