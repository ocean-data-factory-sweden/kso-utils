# base imports
import pandas as pd

# widget imports
from IPython.display import display
import ipywidgets as widgets

# util imports
import kso_utils.db_utils as db_utils
import kso_utils.movie_utils as movie_utils
import kso_utils.spyfish_utils as spyfish_utils


def check_sites_csv(db_initial_info, project):

    # Load the csv with sites information
    sites_df = pd.read_csv(db_initial_info["local_sites_csv"])
    
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
    
    return sites_df

    
def check_movies_csv(db_initial_info, project):

    # Check for missing fps and duration info
    movies_df = movie_utils.check_fps_duration(db_initial_info, project)
    
    # Check if the project is the Spyfish Aotearoa
    if project.Project_name == "Spyfish_Aotearoa":
        movies_df = spyfish_utils.process_spyfish_movies(movies_df)
        
        
    # Check if the project is the KSO
    if project.Project_name == "Koster_Seafloor_Obs":
        movies_df = koster_utils.process_koster_movies_csv(movies_df)
    
    
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
    
    return movies_df
    


def check_movies_from_server(movies_df, sites_df, project, db_info_dict):
    # Get project-specific server info
    server = project.server
    
    if server=="AWS":
        # Retrieve movies that are missing info in the movies.csv
        missing_info = spyfish_utils.check_spyfish_movies(movies_df, db_info_dict)
        
    # Find out files missing from the Server
    missing_from_server = missing_info[missing_info["_merge"]=="right_only"]
    missing_bad_deployment = missing_from_server[missing_from_server["IsBadDeployment"]]
    missing_no_bucket_info = missing_from_server[~(missing_from_server["IsBadDeployment"])]
    
    print("There are", len(missing_from_server.index), "movies missing from", server)
    print(len(missing_bad_deployment.index), "movies are bad deployments. Their filenames are:")
    print(*missing_bad_deployment.filename.unique(), sep = "\n")
    print(len(missing_no_bucket_info.index), "movies are good deployments but don't have movies uploaded. Their filenames are:")
    print(*missing_no_bucket_info.filename.unique(), sep = "\n")
    
    # Find out files missing from the csv
    missing_from_csv = missing_info[missing_info["_merge"]=="left_only"].reset_index(drop=True)
    print("There are", len(missing_from_csv.index), "movies missing from movies.csv. Their filenames are:")
    print(*missing_from_csv.filename.unique(), sep = "\n")
    
    return missing_from_server, missing_from_csv


    
def upload_movies():
    
    # Define widget to upload the files
    mov_to_upload = widgets.FileUpload(
        accept='.mpg',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
        multiple=True  # True to accept multiple files upload else False
    )
    
    # Display the widget?
    display(mov_to_upload)
    
    main_out = widgets.Output()
    display(main_out)
    
    # TODO Copy the movie files to the movies folder
    
    # Provide the site, location, date info of the movies
    upload_info_movies()
    print("uploaded")
    
# Check that videos can be mapped
    movies_df['exists'] = movies_df['Fpath'].map(os.path.isfile)    
    
def upload_info_movies():

    # Select the way to upload the info about the movies
    widgets.ToggleButton(
    value=False,
    description=['I have a csv file with information about the movies',
                 'I am happy to write here all the information about the movies'],
    disabled=False,
    button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Description',
    icon='check'
)
    
    # Upload the information using a csv file
    widgets.FileUpload(
    accept='',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
    multiple=False  # True to accept multiple files upload else False
)
    # Upload the information 
    
    # the folder where the movies are
    
    # Try to extract location and date from the movies 
    widgets.DatePicker(
    description='Pick a Date',
    disabled=False
)
    
    # Run an interactive way to write metadata info about the movies
    
    print("Thanks for providing all the required information about the movies")
    
    
    
    
    
# # Missing info for files in the "buv-zooniverse-uploads"
# missing_info = zoo_contents_s3_pd_movies.merge(movies_df, 
#                                         on=['Key'], 
#                                         how='outer', 
#                                         indicator=True)

# # Find out about those files missing from the S3
# missing_from_s3 = missing_info[missing_info["_merge"]=="right_only"]
# missing_bad_deployment = missing_from_s3[missing_from_s3["IsBadDeployment"]]
# missing_no_bucket_info = missing_from_s3[~(missing_from_s3["IsBadDeployment"])&(missing_from_s3["bucket"].isna())]

# print("There are", len(missing_from_s3.index), "movies missing from the S3")
# print(len(missing_bad_deployment.index), "movies are bad deployments. Their filenames are:")
# print(*missing_bad_deployment.filename.unique(), sep = "\n")
# print(len(missing_no_bucket_info.index), "movies are good deployments but don't have bucket info. Their filenames are:")
# print(*missing_no_bucket_info.filename.unique(), sep = "\n")


# Select multiple movies to include information of
def go_pro_movies_to_update(df):
    
    # Save the filenames of the movies missing
    filename_missing_csv = df.location_and_filename.unique()
    
    # Display the project options
    movie_to_update = widgets.SelectMultiple(
        options=filename_missing_csv,
        rows=15,
        layout=Layout(width='80%'),
        description="GO pro movies:",
        disabled=False,
        
    )
    
    display(movie_to_update)
    return movie_to_update

# Select one movie to include information of
def full_movie_to_update(df):
    
    # Save the filenames of the movies missing
    filename_missing_csv = df.location_and_filename.unique()
    
    # Display the project options
    movie_to_update = widgets.Dropdown(
        options=filename_missing_csv,
        rows=15,
        layout=Layout(width='80%'),
        description="Full movie:",
        disabled=False,
        
    )
    
    display(movie_to_update)
    return movie_to_update


# Select the info to add to the csv
def info_to_csv(df, movies):
    
    # Save the filenames of the movies missing
    filename_missing_csv = df.location_and_filename.unique()
    
    # Display the project options
    movie_to_update = widgets.SelectMultiple(
        options=filename_missing_csv,
        rows=15,
        layout=Layout(width='80%'),
        description="Movie:",
        disabled=False,
        
    )
    
    display(movie_to_update)
    return movie_to_update
    
    