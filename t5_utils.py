#t5 utils
import argparse, os
import kso_utils.db_utils as db_utils
import pandas as pd
import numpy as np
import math
from IPython.display import HTML, display, update_display, clear_output
import ipywidgets as widgets
from ipywidgets import interact
from kso_utils.zooniverse_utils import auth_session

def choose_classes(db_path: str = "koster_lab.db"):
    conn = db_utils.create_connection(db_path)
    species_list = pd.read_sql_query("SELECT label from species", conn)["label"].tolist()
    w = widgets.SelectMultiple(
        options=species_list,
        value=[species_list[0]],
        description='Species',
        disabled=False
    )

    display(w)
    return w

def upload_frames_to_zooniverse(upload_to_zoo, sitename, created_on, project):
    
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

def choose_clip_workflows(workflows_df):

    layout = widgets.Layout(width="auto", height="40px")  # set width and height

    # Display the names of the workflows
    workflow_name = widgets.SelectMultiple(
        options=workflows_df.display_name.unique().tolist(),
        description="Workflow name:",
        disabled=False,
    )

    
    display(workflow_name)

    return workflow_name

def select_workflow(class_df, workflows_df, db_path):
    # Connect to koster_db
    conn = db_utils.create_connection(db_path)

    # Query id and subject type from the subjects table
    subjects_df = pd.read_sql_query("SELECT id, subject_type, https_location, clip_start_time, movie_id FROM subjects WHERE subject_type='clip'", conn)

    # Add subject information based on subject_ids
    class_df = pd.merge(
        class_df,
        subjects_df,
        how="left",
        left_on="subject_ids",
        right_on="id",
    )

    # Select only classifications submitted to clip subjects
    clips_class_df = class_df[class_df.subject_type=='clip']

    # Save the ids of clip workflows
    clip_workflow_ids = class_df.workflow_id.unique()

    # Select clip workflows with classifications
    clip_workflows_df = workflows_df[workflows_df.workflow_id.isin(clip_workflow_ids)]

    # Select the workflows of the video classifications you want to aggregrate
    workflow_names = choose_clip_workflows(clip_workflows_df)
    
    return clips_class_df, workflow_names, workflows_df


def select_workflow_version(w_names, workflows_df):
    
    # Select the workflow ids based on workflow names
    workflow_ids = workflows_df[workflows_df.display_name.isin(w_names)].workflow_id.unique()
    
    # Create empty vector to save the versions selected
    w_versions_list = []

    for w_name in w_names:

        # Estimate the versions of the workflow of interest
        w_versions_available = workflows_df[workflows_df.display_name==w_name].version.unique()

        # Display the versions of the workflow available
        choose_clip_w_version = widgets.Dropdown(
            options=list(map(float, w_versions_available)),
            description= "Min version for " + w_name + ":",
            disabled=False
        )

        # Display a button to select the version
        btn = widgets.Button(description='Select')
        display(choose_clip_w_version, btn)

        def update_version_list(obj):
            print('You have selected',choose_clip_w_version.value)
            w_versions_list = w_versions_list.append(w_name+"_"+choose_clip_w_version.value)
        btn.on_click(update_version_list)

    return w_versions_list