# base imports
import os 
import pandas as pd
import logging

# widget imports
import ipywidgets as widgets
from ipyfilechooser import FileChooser
from IPython.display import HTML

# util imports
import kso_utils.server_utils as server_utils
import kso_utils.db_utils as db_utils
import kso_utils.zooniverse_utils as zooniverse_utils
import kso_utils.project_utils as project_utils


# Logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def choose_folder(start_path: str = ".", folder_type: str = ""):
    # Specify the output folder
    fc = FileChooser(start_path)
    fc.title = f"Choose location of {folder_type}"
    display(fc)
    return fc


def get_project_info(projects_csv, project_name, info_interest):
    
    # Read the latest list of projects
    projects_df = pd.read_csv(projects_csv)
    
    # Get the info_interest from the project info
    project_info = projects_df[projects_df["Project_name"]==project_name][info_interest].unique()[0]
    
    return project_info


def choose_project(projects_csv: str = "../db_starter/projects_list.csv"):
    
    # Check path to the list of projects is a csv
    if os.path.exists(projects_csv) and not projects_csv.endswith(".csv"):
        logging.error("A csv file was not selected. Please try again.")
        
    # If list of projects doesn't exist retrieve it from github
    if not os.path.exists(projects_csv):
        projects_csv = "https://github.com/ocean-data-factory-sweden/koster_data_management/blob/main/db_starter/projects_list.csv?raw=true"
    
    projects_df = pd.read_csv(projects_csv)

    if "Project_name" not in projects_df.columns:
        logging.error("We were unable to find any projects in that file, \
                      please choose a projects csv file that matches our template.")

    # Display the project options
    choose_project = widgets.Dropdown(
        options=projects_df.Project_name.unique().tolist(),
        value=projects_df.Project_name.unique().tolist()[0],
        description="Project:",
        disabled=False,
    )

    display(choose_project)
    return choose_project

def initiate_db(project):
    
    # Get the project-specific name of the database
    db_path = project.db_path
    project_name = project.Project_name
    
    # Initiate the sql db
    db_utils.init_db(db_path)
    
    # Connect to the server (or folder) hosting the csv files
    server_i_dict = server_utils.connect_to_server(project)
    
    # Get the initial info
    db_initial_info = server_utils.get_db_init_info(project, server_i_dict)
    
    # Populate the sites info
    if "local_sites_csv" in db_initial_info.keys():
        db_utils.add_sites(db_initial_info, project_name, db_path)
    
    # Populate the movies info
    if "local_movies_csv" in db_initial_info.keys():
        db_utils.add_movies(db_initial_info, project_name, db_path)
        
    # Populate the photos info
    if "local_photos_csv" in db_initial_info.keys():
        db_utils.add_photos(db_initial_info, project_name, db_path)
    
    # Populate the species info
    if "local_species_csv" in db_initial_info.keys():
        db_utils.add_species(db_initial_info, project_name, db_path)
    
    # Combine server/project info in a dictionary
    db_info_dict = {**db_initial_info, **server_i_dict}
    
    # Add project-specific db_path
    db_info_dict["db_path"] = db_path
    
    return db_info_dict


def connect_zoo_project(project):
    # Save your Zooniverse user name and password.
    zoo_user, zoo_pass = zooniverse_utils.zoo_credentials()
    
    # Get the project-specific zooniverse number
    project_n = project.Zooniverse_number
    
    # Connect to the Zooniverse project
    project = zooniverse_utils.auth_session(zoo_user, zoo_pass, project_n)
    
    return project

def retrieve__populate_zoo_info(project, db_info_dict, zoo_project, zoo_info):
    
    if zoo_project is None:
        logging.error("This project is not linked to a Zooniverse project. Please create one and add the required fields to proceed with this tutorial.")
    else:
        # Retrieve and store the information of subjects uploaded to zooniverse
        zoo_info_dict = zooniverse_utils.retrieve_zoo_info(project, zoo_project, zoo_info)

        # Populate the sql with subjects uploaded to Zooniverse
        zooniverse_utils.populate_subjects(zoo_info_dict["subjects"], 
                                           project,
                                           db_info_dict["db_path"])
        return zoo_info_dict

def choose_single_workflow(workflows_df):

    layout = widgets.Layout(width="auto", height="40px")  # set width and height

    # Display the names of the workflows
    workflow_name = widgets.Dropdown(
        options=workflows_df.display_name.unique().tolist(),
        value=workflows_df.display_name.unique().tolist()[0],
        description="Workflow name:",
        disabled=False,
    )

    # Display the type of subjects
    subj_type = widgets.Dropdown(
        options=["frame", "clip"],
        value="clip",
        description="Subject type:",
        disabled=False,
    )

    display(workflow_name)
    display(subj_type)

    return workflow_name, subj_type

# Function to preview underwater movies
def preview_movie(project, db_info_dict, available_movies_df, movie_i):
    
    # Select the movie of interest
    movie_selected = available_movies_df[available_movies_df["filename"]==movie_i].reset_index(drop=True)

    # Make sure only one movie is selected
    if len(movie_selected.index)>1:
        print("There are several movies with the same filename. This should be fixed!")

    else:
        # Generate temporary path to the movie select
        if project.server == "SNIC":
            movie_path = server_utils.get_movie_url(project, db_info_dict, movie_selected["spath"].values[0])
            os.chdir(os.path.dirname(movie_path))
            return HTML(f"""<video src={os.path.basename(movie_path)} width=800 controls/>"""), movie_path
        else:
            movie_path = server_utils.get_movie_url(project, db_info_dict, movie_selected["fpath"].values[0])
        return HTML(f"""<video src={movie_path} width=800 controls/>"""), movie_path
