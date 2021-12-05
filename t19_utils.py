import os
from ipyfilechooser import FileChooser
from ipywidgets import interactive, Layout, Button, HBox 

import asyncio
import kso_utils.movie_utils as movie_utils
import kso_utils.server_utils as server_utils
import pandas as pd
import ipywidgets as widgets
import numpy as np
import subprocess
import datetime

def record_date_encoder():
    # Widget to record the encoder of the information
    EncoderName_widget = widgets.Text(
        placeholder='First and last name',
        description='Name of the person encoding this BUV information:',
        disabled=False,
        layout=Layout(width='50%'),
        style = {'description_width': 'initial'}
    )
    
    # Widget to record the dateEntry of the information
    DateEntry_widget = widgets.DatePicker(
        description='Entry date for the new information (aka Today)',
        disabled=False,
        layout=Layout(width='50%'),
        style = {'description_width': 'initial'}
    )


    display(EncoderName_widget, DateEntry_widget)
    
    return EncoderName_widget, DateEntry_widget
    
            
            
def select_go_pro_folder():
    # Create and display a FileChooser widget
    fc = FileChooser('/')
    
    display(fc)
    
    return fc

def select_go_pro_movies(go_pro_folder):
    # Save the names of the go_pro files
    go_pro_files_i = [go_pro_folder + movie for movie in os.listdir(go_pro_folder)]
    
    # Specify the formats of the movies to select
    movie_formats = movie_utils.get_movie_extensions()
    
    # Select only movie files
    go_pro_movies_i = [s for s in go_pro_files_i if any(xs in s for xs in movie_formats)]
    
    print("The movies selected are:")
    print(*go_pro_movies_i, sep='\n')

    return go_pro_movies_i
    

# Select site and date of the video
def select_sitename(db_initial_info):
    
    # Read csv as pd
    sitesdf = pd.read_csv(db_initial_info["local_sites_csv"])

    # Existing sites
    exisiting_sites = sitesdf.sort_values("SiteName").SiteName.unique()
    
    site_widget = widgets.Dropdown(
                options = exisiting_sites,
                description = 'Site:',
                disabled = False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
    
    
    display(site_widget)

    return site_widget


def select_eventdate():
    
    # Select the date 
    date_widget = widgets.DatePicker(
        description='Deployment or event date:',
        disabled=False,
        layout=Layout(width='50%'),
        style = {'description_width': 'initial'}
    )
    
    
    display(date_widget)
    return date_widget  
    

# Function to download go pro videos, concatenate them and upload the concatenated videos to aws 
def concatenate_go_pro_videos(SiteName_i, EventDate_i, go_pro_folder, go_pro_files_i):

    # Specify the name of the deployment
    deployment_name_i = SiteName_i+"_"+EventDate_i

    # Specify temp folder to host the concat video
    concat_folder = go_pro_folder+"concat_video/"
   
    # Specify the filename and path for the concatenated movie
    filename_i = deployment_name_i+".MP4"
    concat_video = concat_folder+filename_i

    # Save list as text file
    textfile = open("a_file.txt", "w")
    for go_pro_file in go_pro_files_i:
        textfile.write("file '"+ go_pro_file + "'"+ "\n")
    textfile.close()

    
    if not os.path.exists(concat_folder):
        os.mkdir(concat_folder)
    
    if not os.path.exists(concat_video):
        print("Concatenating ", concat_video)
        
        # Concatenate the videos
        subprocess.call(["ffmpeg",
                         "-f", "concat", 
                         "-safe", "0",
                         "-i", "a_file.txt", 
                         "-c", "copy",
                         concat_video])
            
        print(concat_video, "concatenated successfully")
        
    # Delete the text file
    os.remove("a_file.txt")
        
    # Update the fps and length info
    fps_i, duration_i = movie_utils.get_length(concat_video)
    
    video_info_dict = {
        "fps_i": fps_i, 
        "duration_i": duration_i, 
        "concat_video_i": concat_video, 
        "filename_i": filename_i, 
        "SiteName_i": SiteName_i, 
        "EventDate_i": EventDate_i, 
        "go_pro_files_i": go_pro_files_i, 
        "deployment_name_i": deployment_name_i
    }
    
    print("Open", video_info_dict["concat_video_i"], "to complete the next steps.")
    
    return video_info_dict

def record_deployment_info(db_info_dict, video_info_dict):
    
    # Read csv as pd
    movies_df = pd.read_csv(db_info_dict["local_movies_csv"])
    
    # Write the link to the fieldsheets
    LinkToFieldSheets_i = select_LinkToFieldSheets()
    
    # Select the estimated length of deployment
    DeploymentDurationMinutes_i = select_DeploymentDurationMinutes()
    
    # Select the time in
    TimeIn_i = deployment_TimeIn()
    
    # Select the time out
    TimeOut_i = deployment_TimeOut()
    
    # Select author
    RecordedBy_i = select_author(movies_df)
    
    # Select depth stratum of the deployment
    DepthStrata_i = select_DepthStrata()
    
    # Select depth of the deployment
    Depth_i = deployment_depth()
    
    # Select the underwater visibility
    UnderwaterVisibility_i = select_UnderwaterVisibility()

    # Specify id deployment is bad
    IsBadDeployment_i = select_bad_deployment()

    # Set survey start
    SurveyStart_i = select_start_survey(video_info_dict["duration_i"])

    #TODO Set survey end
    SurveyEnd_i = select_end_survey(video_info_dict["duration_i"])

    # Select the S3 "survey folder" to upload the video
    s3_folder_i = select_s3_folder(db_info_dict)

    #Add any comment related to the movie
    NotesDeployment_i = write_comment()
    
    deployment_info = {
        "RecordedBy_i": RecordedBy_i,
        "IsBadDeployment_i": IsBadDeployment_i,
        "SurveyStart_i": SurveyStart_i,
        "SurveyEnd_i": SurveyEnd_i,
        "LinkToFieldSheets_i": LinkToFieldSheets_i,
        "s3_folder_i": s3_folder_i,
        "NotesDeployment_i": NotesDeployment_i,
        "TimeIn_i": TimeIn_i,
        "TimeOut_i":TimeOut_i,
        "Depth_i": Depth_i,
        "DepthStrata_i": DepthStrata_i,
        "UnderwaterVisibility_i": UnderwaterVisibility_i,
        "DeploymentDurationMinutes_i": DeploymentDurationMinutes_i,
    }

    return deployment_info

def select_DeploymentDurationMinutes():
    # Select the depth of the deployment 
    DeploymentDurationMinutes = widgets.BoundedIntText(
                value=0,
                min=0,
                max=60,
                step=1,
                description='Theoretical minimum soaking time for the unit (mins):',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
    
    display(DeploymentDurationMinutes)

    return DeploymentDurationMinutes
  
    
    
def select_LinkToFieldSheets():
    LinkToFieldSheets = widgets.Text(
                description='Hyperlink to the DOCCM for the field sheets used to gather the information on BUV deployment:',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
    
    display(LinkToFieldSheets)
    return LinkToFieldSheets

def deployment_TimeIn():
    
    # Select the TimeIn
    TimeIn_widget = widgets.TimePicker(
        description='Time in of the deployment:',
        disabled=False,
        layout=Layout(width='50%'),
        style = {'description_width': 'initial'}
    )
   
    display(TimeIn_widget)
    return TimeIn_widget  

def deployment_TimeOut():
    
    # Select the TimeOut
    TimeOut_widget = widgets.TimePicker(
        description='Time out of the deployment:',
        disabled=False,
        layout=Layout(width='50%'),
        style = {'description_width': 'initial'}
    )
    
    
    display(TimeOut_widget)
    return TimeOut_widget  


def select_UnderwaterVisibility():
    
    UnderwaterVisibility = widgets.Dropdown(
                        options = ['Good','Fair','Poor'],
                        description = 'Water visibility of the video deployment:',
                        disabled = False,
                        layout=Layout(width='50%'),
                        style = {'description_width': 'initial'}
                    )
    
    
    display(UnderwaterVisibility)
    return UnderwaterVisibility
    


# Select author of the video
def select_author(movies_df):
    
    # Existing authors
    exisiting_authors = movies_df.Author.unique()

    def f(Existing_or_new):
        if Existing_or_new == 'Existing':
            author_widget = widgets.Dropdown(
                options = exisiting_authors,
                description = 'Existing author:',
                disabled = False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )

        if Existing_or_new == 'New author':   
            author_widget = widgets.Text(
                placeholder='First and last name',
                description='Author:',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )

        display(author_widget)

        return(author_widget)

    w = interactive(f,
                    Existing_or_new = widgets.Dropdown(
                        options = ['Existing','New author'],
                        description = 'Deployment recorded by existing or new author:',
                        disabled = False,
                        layout=Layout(width='50%'),
                        style = {'description_width': 'initial'}
                    )
                   )

    display(w)

    return w
    
def select_DepthStrata():
    # Select the depth of the deployment 
    deployment_DepthStrata = widgets.Text(
                placeholder='5-25m',
                description='Depth stratum within which the BUV unit was deployed:',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
    
    display(deployment_DepthStrata)

    return deployment_DepthStrata

    
    
def deployment_depth():
    # Select the depth of the deployment 
    deployment_depth = widgets.BoundedIntText(
                value=0,
                min=0,
                max=100,
                step=1,
                description='Depth reading in meters at the time of BUV unit deployment:',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
    
    display(deployment_depth)

    return deployment_depth
  
      

def select_bad_deployment():
    
    def deployment_to_true_false(deploy_value):
        if deploy_value == 'No, it is a great video':
            return False
        else:
            return True

    w = interactive(deployment_to_true_false, deploy_value = widgets.Dropdown(
        options=['Yes, unfortunately it is marine crap', 'No, it is a great video'],
            value='No, it is a great video',
            description='Is it a bad deployment?',
            disabled=False,
            layout=Layout(width='50%'),
            style = {'description_width': 'initial'}
        ))

    display(w)

    return w
    
    
    
# Display in hours, minutes and seconds
def to_hhmmss(seconds):
    print("Time selected:", datetime.timedelta(seconds=seconds))
    
    return seconds

def select_start_survey(duration_i):

    # Select the start of the survey 
    surv_start = interactive(to_hhmmss, seconds=widgets.IntSlider(
        value=0,
        min=0,
        max=duration_i,
        step=1,
        description='Survey starts (seconds):',
        layout=Layout(width='50%'),
        style = {'description_width': 'initial'}))

    display(surv_start)    
    
    return surv_start
 
    
def select_end_survey(duration_i):
    
#     # Set default to 30 mins or max duration
#     start_plus_30 = surv_start_i+(30*60)
    
#     if start_plus_30>duration_i:
#         default_end = duration_i
#     else:
#         default_end = start_plus_30
    
    
    # Select the end of the survey 
    surv_end = interactive(to_hhmmss, seconds=widgets.IntSlider(
        value=duration_i,
        min=0,
        max=duration_i,
        step=1,
        description='Survey ends (seconds):',
        layout=Layout(width='50%'),
        style = {'description_width': 'initial'}))

    display(surv_end)  
    
    return surv_end
    
# Select s3 folder to upload the video
def select_s3_folder(db_info_dict):

    # Specify the bucket
    bucket_i = 'marine-buv'
    
    # Retrieve info from the bucket
    contents_s3_pd = server_utils.get_matching_s3_keys(db_info_dict["client"], bucket_i, "")

    # Extract the prefix (directory) of the objects        
    s3_folders_available = contents_s3_pd["Key"].str.split("/").str[0]

    # Select the s3 folder
    s3_folder_widget = widgets.Combobox(
                    options=tuple(s3_folders_available.unique()),
                    description="S3 folder:",
                    ensure_option=True,
                    disabled=False,
                )
    
    
    display(s3_folder_widget)
    return s3_folder_widget

# Write a comment about the video
def write_comment():

    # Create the comment widget
    comment_widget = widgets.Text(
            placeholder='Type comment',
            description='Comment:',
            disabled=False,
            layout=Layout(width='50%'),
            style = {'description_width': 'initial'}
        )

    
    display(comment_widget)
    return comment_widget


def confirm_deployment_details(video_info_dict_i,
                               db_info_dict,
                               survey_i,
                               deployment_info,
                               date_encoder,
                               survey_name):
    
    # Read movies csv
    movies_df = pd.read_csv(db_info_dict["local_movies_csv"])
    
    # Add movie id
    movie_id_i = 1 + movies_df.movie_id.iloc[-1]
    
    # Save the name of the survey
    if isinstance(survey_i.result, tuple):
        
        # Load the csv with survey information
        surveys_df = pd.read_csv(db_info_dict["local_surveys_csv"])
            
        # Save the latest survey added
        survey_name = surveys_df.tail(1)["SurveyName"].values[0]
   
    else:
        # Return the name of the survey
        survey_name = survey_i.result.value
            
            
    # Create temporary prefix (s3 path) for concatenated video
    prefix_conc_i = deployment_info("s3_folder_i") + "/" + video_info_dict_i["deployment_name_i"] + "/" + video_info_dict_i["filename_i"]
    
    # Read surveys csv
    surveys_df = pd.read_csv(db_info_dict["local_surveys_csv"])
    
    # Select relevant survey
    surveys_df_i = surveys_df[surveys_df["SurveyName"]==survey_name].reset_index() 
    
    # Select previously processed movies within the same survey
    survey_movies_df = movies_df[movies_df["SurveyID"]==surveys_df_i["SurveyID"].values[0]].reset_index()
    
    # Create unit id
    if not survey_movies_df:
        # Start unit_id in 0
        UnitID = surveys_df_i["SurveyID"].values[0] + "_0000"
        
    else:
        # Get the last unitID
        last_unitID = str(survey_movies_df.sort("UnitID").tail(1).values[0])[-4:]
        
        # Add one more to the last UnitID
        next_unitID = str(int(last_unitID) + 1).zfill(4)
        
        # Add one more to the last UnitID
        UnitID = surveys_df_i["SurveyID"].values[0] + "_" + next_unitID
        
    # Save the responses as a new row for the survey csv file
    new_deployment_row = pd.DataFrame(
        {
            "movie_id": movie_id_i,
            "RecordedBy": deployment_info["RecordedBy_i"],
            "DateEntryMovie": date_encoder[1].value,
            "EncoderNameMovie": date_encoder[0].value,
            "EventDate": video_info_dict_i["EventDate_i"],
            "SurveyStart": deployment_info["SurveyStart_i"],
            "SurveyEnd": deployment_info["SurveyEnd_i"],
            "prefix_conc": prefix_conc_i,
            "filename": video_info_dict_i["filename_i"],
            "fps": video_info_dict_i["fps_i"],
            "duration": video_info_dict_i["duration_i"],
            "go_pro_files": video_info_dict_i["go_pro_files_i"],
            "IsBadDeployment": deployment_info["IsBadDeployment_i"],
            "LinkToFieldSheets": deployment_info["LinkToFieldSheets_i"],
            "LinkReport01": "NA",
            "LinkReport02": "NA",
            "LinkReport03": "NA",
            "LinkReport04": "NA",
            "LinkToOriginalData": "NA",
            "UnitID": UnitID,
            "ReplicateWithinSite": "NA",
            "RecordedBy": deployment_info["RecordedBy_i"],
            "Year": video_info_dict_i["EventDate_i"].year,
            "Month": video_info_dict_i["EventDate_i"].month,
            "Day": video_info_dict_i["EventDate_i"].day,
            "DepthStrata": deployment_info["DepthStrata_i"],
            "Depth": deployment_info["Depth_i"],
            "UnderwaterVisibility": deployment_info["UnderwaterVisibility_i"],
            "TimeIn": deployment_info["TimeIn_i"],
            "TimeOut": deployment_info["TimeOut_i"],
            "NotesDeployment": deployment_info["NotesDeployment_i"],
            "DeploymentDurationMinutes": deployment_info["DeploymentDurationMinutes_i"],
            "SurveyID": surveys_df_i["SurveyID"].values[0],
            "SurveyName": survey_name,
        }
    )
    
    print("The details of the new deployment are:")
    for ind in new_deployment_row.T.index:
        print(ind,"-->", new_deployment_row.T[0][ind])
    
    return new_deployment_row


def upload_concat_movie(db_info_dict, new_deployment_row):
    
    # Upload movie to the s3 bucket
    server_utils.upload_file_to_s3(client = db_info_dict["client"],
                                   bucket = db_info_dict["bucket"], 
                                   key = new_deployment_row["prefix_conc"], 
                                   filename = video_info_dict["concat_video"])
    
    # Create the link to concat video file in s3
    location = db_info_dict["client"].get_bucket_location(Bucket=db_info_dict["bucket"])['LocationConstraint']
    url = "https://s3-%s.amazonaws.com/%s/%s" % (location, db_info_dict["bucket"], new_deployment_row["prefix_conc"])
    
    # Save to new deployment row df
    new_deployment_row["LinkToVideoFile"] = url
    
    print("Movie uploaded to", new_deployment_row["LinkToVideoFile"])
    
    # Remove temporary prefix for concatenated video
    new_deployment_row = new_deployment_row.drop("prefix_conc")
    
    # Load the csv with movies information
    movies_df = pd.read_csv(db_info_dict["local_movies_csv"])

    # Add the new row to the movies df
    movies_df = movies_df.append(new_deployment_row, ignore_index=True)

    # Save the updated df locally
    movies_df.to_csv(db_info_dict["local_movies_csv"],index=False)

    # Save the updated df in the server
    server_utils.upload_file_to_s3(db_info_dict["client"],
                                   bucket=db_info_dict["bucket"], 
                                   key=db_info_dict["server_movies_csv"], 
                                   filename=db_info_dict["local_movies_csv"])
    
    # Remove temporary movie
    
    print("Movies csv file succesfully updated")
    

def select_survey(db_info_dict):
    # Load the csv with surveys information
    surveys_df = pd.read_csv(db_info_dict["local_surveys_csv"])
    
    # Existing Surveys
    exisiting_surveys = surveys_df.SurveyName.unique()

    def f(Existing_or_new):
        if Existing_or_new == 'Existing':
            survey_widget = widgets.Dropdown(
                options = exisiting_surveys,
                description = 'Survey Name:',
                disabled = False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
            
            display(survey_widget)

            return(survey_widget)

        if Existing_or_new == 'New survey':   
            
            # Load the csv with with sites and survey choices
            choices_df = pd.read_csv(db_info_dict["local_choices_csv"])
            
            # Widget to record the start date of the survey
            SurveyStartDate_widget = widgets.DatePicker(
                description='Offical date when survey started as a research event',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )

            # Widget to record the name of the survey
            SurveyName_widget = widgets.Text(
                placeholder='Baited Underwater Video Taputeranga Apr 2015',
                description='A name for this survey:',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
            
            # Widget to record the 3 letter abbreviation for the Marine Reserve
            SurveyLocation_widget = widgets.Text(
                placeholder='YYY',
                description='The official 3 letter abbreviation for the Marine Reserve:',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
            
            
            # Widget to record the type of survey
            SurveyType_widget = widgets.Dropdown(
                        options = ['BUV','ROV'],
                        value='BUV',
                        description = 'Type of survey:',
                        disabled = False,
                        layout=Layout(width='50%'),
                        style = {'description_width': 'initial'}
            )

            # Widget to record the name of the contractor
            ContractorName_widget = widgets.Text(
                placeholder='No contractor',
                description='Person/company contracted to carry out the survey:',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
            
            # Widget to record the number of the contractor
            ContractNumber_widget = widgets.Text(
                description='Contract number for this survey:',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
            
            # Widget to record the link to the contract
            LinkToContract_widget = widgets.Text(
                description='Hyperlink to the DOCCM for the contract related to this survey:',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
            
            # Widget to record the name of the survey leader
            SurveyLeaderName_widget = widgets.Text(
                placeholder='First and last name',
                description='Name of the person in charge of this survey:',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
            
            # Widget to record the name of the linked Marine Reserve
            LinkToMarineReserve_widget = widgets.Dropdown(
                options=choices_df.MarineReserve.dropna().unique().tolist(),
                description='Marine Reserve linked to the survey:',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
            
            # Widget to record if survey is single species
            FishMultiSpecies_widget = widgets.Dropdown(
                options=["No", "Yes"],
                description='Does this survey look at a single species?',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )

            # Widget to record if survey was stratified by any factor
            StratifiedBy_widget = widgets.Dropdown(
                options=choices_df.Stratification.dropna().unique().tolist(),
                description='Stratified factors for the sampling design',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )

            # Widget to record if survey is part of long term monitoring
            IsLongTermMonitoring_widget = widgets.Dropdown(
                options=["Yes", "No"],
                description='Is the survey part of a long-term monitoring?',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )

            # Widget to record the site selection of the survey
            SiteSelectionDesign_widget = widgets.Dropdown(
                options=choices_df.SiteSelection.dropna().unique().tolist(),
                description='What was the design for site selection?',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )

            # Widget to record the unit selection of the survey
            UnitSelectionDesign_widget = widgets.Dropdown(
                options=choices_df.UnitSelection.dropna().unique().tolist(),
                description='What was the design for site selection?',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
            
            # Widget to record the type of right holder of the survey
            RightsHolder_widget = widgets.Dropdown(
                options=choices_df.RightsHolder.dropna().unique().tolist(),
                description='Person(s) or organization(s) owning or managing rights over the resource',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )

            # Widget to record information about who can access the resource
            AccessRights_widget = widgets.Text(
                placeholder='',
                description='Who can access the resource?',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
            
            # Widget to record description of the survey design and objectives
            SurveyVerbatim_widget = widgets.Textarea(
                placeholder='',
                description='Provide an exhaustive description of the survey design and objectives',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )

            # Widget to record the type of BUV
            BUVType_widget = widgets.Dropdown(
                options=choices_df.BUVType.dropna().unique().tolist(),
                description='Type of BUV used for the survey:',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
            
            
            # Widget to record the type of camera
            CameraModel_widget = widgets.Text(
                placeholder='Make and model',
                description='Describe the type of camera used',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
            
            # Widget to record the camera settings
            CameraSettings_widget = widgets.Text(
                placeholder='Wide lens, 1080x1440',
                description='Describe the camera settings',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )

            # Widget to record the type of bait used
            BaitSpecies_widget = widgets.Text(
                placeholder='Pilchard',
                description='Species that was used as bait for the deployment',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
            
            
            # Widget to record the amount of bait used
            BaitAmount_widget = widgets.BoundedIntText(
                value=500,
                min=100,
                max=1000,
                step=1,
                description='Amount of bait used (g):',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
            
            # Widget to record the link to the pictures
            LinkToPicture_widget = widgets.Text(
                description='Hyperlink to the DOCCM folder for this survey photos:',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )

            # Widget to record the name of the vessel
            Vessel_widget = widgets.Text(
                description='Vessel used to deploy the unit:',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
            
            # Widget to record the level of the tide
            TideLevel_widget = widgets.Dropdown(
                options=choices_df.TideLevel.dropna().unique().tolist(),
                description='Tidal level at the time of sampling:',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
            
            # Widget to record the weather
            Weather_widget = widgets.Text(
                description='Describe the weather for the survey:',
                disabled=False,
                layout=Layout(width='50%'),
                style = {'description_width': 'initial'}
            )
            


            
            display(SurveyStartDate_widget,
                    SurveyName_widget,
                   SurveyLocation_widget,
                   SurveyType_widget,
                   ContractorName_widget,
                   ContractNumber_widget,
                   LinkToContract_widget,
                   SurveyLeaderName_widget,
                   LinkToMarineReserve_widget,
                   FishMultiSpecies_widget,
                   StratifiedBy_widget,
                   IsLongTermMonitoring_widget,
                   SiteSelectionDesign_widget,
                   UnitSelectionDesign_widget,
                   RightsHolder_widget,
                   AccessRights_widget,
                   SurveyVerbatim_widget,
                   BUVType_widget,
                   CameraModel_widget,
                   CameraSettings_widget,
                   BaitSpecies_widget,
                   BaitAmount_widget,
                   LinkToPicture_widget,
                   Vessel_widget,
                   TideLevel_widget,
                   Weather_widget)

            
            return(SurveyStartDate_widget,
                  SurveyName_widget,
                  SurveyLocation_widget,
                  SurveyType_widget,
                  ContractorName_widget,
                  ContractNumber_widget,
                  LinkToContract_widget,
                  SurveyLeaderName_widget,
                  LinkToMarineReserve_widget,
                  FishMultiSpecies_widget,
                  StratifiedBy_widget,
                  IsLongTermMonitoring_widget,
                  SiteSelectionDesign_widget,
                  UnitSelectionDesign_widget,
                  RightsHolder_widget,
                  AccessRights_widget,
                  SurveyVerbatim_widget,
                  BUVType_widget,
                  CameraModel_widget,
                  CameraSettings_widget,
                  BaitSpecies_widget,
                  BaitAmount_widget,
                  LinkToPicture_widget,
                  Vessel_widget,
                  TideLevel_widget,
                  Weather_widget)

    w = interactive(f,
                    Existing_or_new = widgets.Dropdown(
                        options = ['Existing','New survey'],
                        description = 'Existing or new survey:',
                        disabled = False,
                        layout=Layout(width='50%'),
                        style = {'description_width': 'initial'}
                    )
                   )

    display(w)

    return w


def wait_for_change(widget1, widget2): #<------ Rename to widget1, and add widget2
    future = asyncio.Future()
    def getvalue(change):
        future.set_result(change.description)
        widget1.on_click(getvalue, remove=True) #<------ Rename to widget1
        widget2.on_click(getvalue, remove=True) #<------ New widget2
        # we need to free up the binding to getvalue to avoid an IvalidState error
        # buttons don't support unobserve
        # so use `remove=True` 
    widget1.on_click(getvalue) #<------ Rename to widget1
    widget2.on_click(getvalue) #<------ New widget2
    return future


# 
def confirm_survey(survey_i, db_info_dict, date_encoder):

    correct_button = widgets.Button(
        description = 'Yes, details are correct',
        layout=Layout(width='25%'),
        style = {'description_width': 'initial'},
        button_style='danger'
        )

    wrong_button = widgets.Button(
        description = 'No, I will go back and fix them',
        layout=Layout(width='45%'),
        style = {'description_width': 'initial'}, 
        button_style='danger'
    )


    # If new survey, review details and save changes in survey csv server
    if isinstance(survey_i.result, tuple):
        # Save the responses as a new row for the survey csv file
        new_survey_row = pd.DataFrame(
            {
                "DateEntrySurvey": [date_encoder[1].value],
                "EncoderNameSurvey": [date_encoder[0].value],
                "SurveyStartDate": [survey_i.result[0].value],
                "SurveyName": [survey_i.result[1].value],
                "SurveyLocation": [survey_i.result[2].value],
                "SurveyType": [survey_i.result[3].value],
                "ContractorName": [survey_i.result[4].value],
                "ContractNumber": [survey_i.result[5].value],
                "LinkToContract": [survey_i.result[6].value],
                "SurveyLeaderName": [survey_i.result[7].value],
                "LinkToMarineReserve": [survey_i.result[8].value],
                "FishMultiSpecies": [survey_i.result[9].value],
                "StratifiedBy": [survey_i.result[10].value],
                "IsLongTermMonitoring": [survey_i.result[11].value],
                "SiteSelectionDesign": [survey_i.result[12].value],
                "UnitSelectionDesign": [survey_i.result[13].value],
                "RightsHolder": [survey_i.result[14].value],
                "AccessRights": [survey_i.result[15].value],
                "SurveyVerbatim": [survey_i.result[16].value],
                "BUVType": [survey_i.result[17].value],
                "CameraModel": [survey_i.result[18].value],
                "CameraSettings": [survey_i.result[19].value],
                "BaitSpecies": [survey_i.result[20].value],
                "BaitAmount": [survey_i.result[21].value],
                "LinkToPicture": [survey_i.result[22].value],
                "Vessel": [survey_i.result[23].value],
                "TideLevel": [survey_i.result[24].value],
                "Weather": [survey_i.result[25].value],
            }
        )

        # Create new columns for the survey based on the responses
        new_survey_row["SurveyID"] = new_survey_row["SurveyLocation"]+ "_" + new_survey_row["SurveyStartDate"].values[0].strftime("%Y%m%d") + "_" + new_survey_row["SurveyType"]
        new_survey_row["FishMultiSpecies"] = new_survey_row.replace({'FishMultiSpecies': {'Yes': False, 'No': True}})
        new_survey_row["IsLongTermMonitoring"] = new_survey_row.replace({'IsLongTermMonitoring': {'Yes': True, 'No': False}})
        new_survey_row[["Region","OfficeName","OfficeContact","MarineReserveID"]] = "NA"

        print("The details of the new survey are:")
        for ind in new_survey_row.T.index:
            print(ind,"-->", new_survey_row.T[0][ind])

        async def f():
            x = await wait_for_change(correct_button,wrong_button) #<---- Pass both buttons into the function
            if x == "Yes, details are correct": #<--- use if statement to trigger different events for the two buttons
                print("Updating the new survey information.")
                
                # Load the csv with sites information
                surveys_df = pd.read_csv(db_info_dict["local_surveys_csv"])
                
                # Add the new row to the choices df
                surveys_df = surveys_df.append(new_survey_row, ignore_index=True)
                
                # Save the updated df locally
                surveys_df.to_csv(db_info_dict["local_surveys_csv"],index=False)
               
                # Save the updated df in the server
                server_utils.upload_file_to_s3(db_info_dict["client"],
                                               bucket=db_info_dict["bucket"], 
                                               key=db_info_dict["server_surveys_csv"], 
                                               filename=db_info_dict["local_surveys_csv"])
                
                print("Survey information updated!")
                
            else:
                print("Come back when the data is tidy!")


    # If existing survey print the info for the pre-existing survey
    else:
        # Load the csv with surveys information
        surveys_df = pd.read_csv(db_info_dict["local_surveys_csv"])

        # Select the specific survey info
        surveys_df_i = surveys_df[surveys_df["SurveyName"]==survey_i.result.value].reset_index(drop=True)

        print("The details of the selected survey are:")
        for ind in surveys_df_i.T.index:
            print(ind,"-->", surveys_df_i.T[0][ind])

        async def f():
            x = await wait_for_change(correct_button,wrong_button) #<---- Pass both buttons into the function
            if x == "Yes, details are correct": #<--- use if statement to trigger different events for the two buttons
                print("Great, you can start uploading the movies.")
                
            else:
                print("Come back when the data is tidy!")

    print("")
    print("")
    print("Are the survey details above correct?")
    display(HBox([correct_button,wrong_button])) #<----Display both buttons in an HBox
    asyncio.create_task(f())
    


