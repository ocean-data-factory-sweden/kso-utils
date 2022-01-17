import os, csv
import pandas as pd
import numpy as np
from pathlib import Path



def process_sgu_photos_csv(db_initial_info):
    # Load the csv with photos and survey information
    photos_df = pd.read_csv(db_initial_info["local_photos_csv"])
    surveys_df = pd.read_csv(db_initial_info["local_surveys_csv"])
    
    # Add survey info to the photos information
    photos_df = photos_df.merge(surveys_df.rename(columns = {"ID": "SurveyID"}),
                                on= "SurveyID",
                                how='left')
    
    # TO DO Include server's path to the photo files
    photos_df["fpath"] = photos_df["filename"]
    
    # Rename to match schema format
    photos_df = photos_df.rename(
        columns = {
            "SiteID": "site_id",# site id for the db
            "SurveyDate": "created_on",
            
        }
    )
          
    return photos_df
    
