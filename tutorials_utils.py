import pandas as pd
import ipywidgets as widgets
import utils.server_utils as server_utils
import utils.db_utils as db_utils
import utils.zooniverse_utils as zooniverse_utils
import panoptes_client


def get_project_info(project_name, info_interest):
    # Specify location of the latest list of projects
    projects_csv = "../db_starter/projects_list.csv" 
    
    # Read the latest list of projects
    projects_df = pd.read_csv(projects_csv)
    
    # Get the info_interest from the project info
    project_info = projects_df[projects_df["Project_name"]==project_name][info_interest].unique()[0]
    
    return project_info

    
def choose_project():
    
    # Specify location of the latest list of projects
    projects_csv = "../db_starter/projects_list.csv" 
    
    # Read the latest list of projects
    projects_df = pd.read_csv(projects_csv)
    
    # Display the project options
    choose_project = widgets.Dropdown(
        options=projects_df.Project_name.unique().tolist(),
        value=projects_df.Project_name.unique().tolist()[0],
        description="Project:",
        disabled=False,
    )
    
    display(choose_project)
    return choose_project

def initiate_db(project_name):
    
    # Get the project-specific name of the database
    db_path = get_project_info(project_name, "db_path")
    
    # Initiate the sql db
    db_utils.init_db(db_path)
    
    # Connect to the server (or folder) hosting the csv files
    server_i_dict = server_utils.connect_to_server(project_name)
    
    # Get the initial info
    db_initial_info = server_utils.get_db_init_info(project_name, server_i_dict)
    
    # Populate the sites info 
    db_utils.add_sites(db_initial_info["sites_csv"], db_path)
    
    # Populate the movies info
    db_utils.add_movies(db_initial_info["movies_csv"], project_name, db_path)
    
    # Populate the species info
    db_utils.add_species(db_initial_info["species_csv"], db_path)
    
    # Combine server/project info in a dictionary
    db_info_dict = {**db_initial_info, **server_i_dict}
    
    # Add project-specific db_path
    db_info_dict["db_path"] = db_path
    
    return db_info_dict
    
    
def connect_zoo_project(project_name):
    # Save your Zooniverse user name and password.
    zoo_user, zoo_pass = zooniverse_utils.zoo_credentials()
    
    # Get the project-specific zooniverse number
    project_n = get_project_info(project_name, "Zooniverse_number")
    
    # Connect to the Zooniverse project
    project = zooniverse_utils.auth_session(zoo_user, zoo_pass, project_n)
    
    return project
    
def retrieve__populate_zoo_info(project_name, db_info_dict, zoo_project, zoo_info):
    
    # Retrieve and store the information of subjects uploaded to zooniverse
    zoo_info_dict = zooniverse_utils.retrieve_zoo_info(project_name, zoo_project, zoo_info)
        
    # Populate the sql with subjects uploaded to Zooniverse
    zooniverse_utils.populate_subjects(zoo_info_dict["subjects"], 
                                       project_name,
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

# def choose_clip_workflows(workflows_df):

#     layout = widgets.Layout(width="auto", height="40px")  # set width and height

#     # Display the names of the workflows
#     workflow_name = widgets.SelectMultiple(
#         options=workflows_df.display_name.unique().tolist(),
#         description="Workflow name:",
#         disabled=False,
#     )

    
#     display(workflow_name)

#     return workflow_name