# base imports
import pandas as pd
import logging

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def process_sgu_photos_csv(db_initial_info: dict):
    """
    It takes the local csv files with photos and surveys information and returns a dataframe with the
    photos information

    :param db_initial_info: a dictionary with the following keys:
    :return: A dataframe with the photos information
    """
    # Load the csv with photos and survey information
    photos_df = pd.read_csv(db_initial_info["local_photos_csv"])
    surveys_df = pd.read_csv(db_initial_info["local_surveys_csv"])

    # Add survey info to the photos information
    photos_df = photos_df.merge(
        surveys_df.rename(columns={"ID": "SurveyID"}), on="SurveyID", how="left"
    )

    # TO DO Include server's path to the photo files
    photos_df["fpath"] = photos_df["filename"]

    # Rename to match schema format
    photos_df = photos_df.rename(
        columns={
            "SiteID": "site_id",  # site id for the db
            "SurveyDate": "created_on",
        }
    )

    return photos_df
