# base imports
import os
import pandas as pd
import subprocess
import datetime
import logging
import ffmpeg
import shutil
from pathlib import Path

# widget imports
import ipywidgets as widgets
from ipywidgets import interactive, Layout, HBox
from IPython.display import display
import asyncio
import ipysheet
from ipyfilechooser import FileChooser

# util imports
import kso_utils.movie_utils as movie_utils
import kso_utils.server_utils as server_utils
import kso_utils.tutorials_utils as t_utils
import kso_utils.project_utils as p_utils


# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


####################################################
############### SURVEY FUNCTIONS ###################
####################################################


def progress_handler(progress_info):
    print("{:.2f}".format(progress_info["percentage"]))


def select_survey(db_info_dict: dict):
    """
    This function allows the user to select an existing survey from a dropdown menu or create a new
    survey by filling out a series of widgets

    :param db_info_dict: a dictionary with the following keys:
    :type db_info_dict: dict
    :return: A widget with a dropdown menu with two options: 'Existing' and 'New survey'.
    """
    # Load the csv with surveys information
    surveys_df = pd.read_csv(db_info_dict["local_surveys_csv"])

    # Existing Surveys
    exisiting_surveys = surveys_df.SurveyName.unique()

    def f(Existing_or_new):
        if Existing_or_new == "Existing":
            survey_widget = widgets.Dropdown(
                options=exisiting_surveys,
                description="Survey Name:",
                disabled=False,
                layout=Layout(width="80%"),
                style={"description_width": "initial"},
            )

            display(survey_widget)

            return survey_widget

        if Existing_or_new == "New survey":
            # Load the csv with with sites and survey choices
            choices_df = pd.read_csv(db_info_dict["local_choices_csv"])

            # Save the new survey responses into a dict
            survey_info = {
                # Write the name of the encoder
                "EncoderName": record_encoder(),
                # Select the start date of the survey
                "SurveyStartDate": select_SurveyStartDate(),
                # Write the name of the survey
                "SurveyName": write_SurveyName(),
                # Select the DOC office
                "OfficeName": select_OfficeName(
                    choices_df.OfficeName.dropna().unique().tolist()
                ),
                # Write the name of the contractor
                "ContractorName": write_ContractorName(),
                # Write the number of the contractor
                "ContractNumber": write_ContractNumber(),
                # Write the link to the contract
                "LinkToContract": write_LinkToContract(),
                # Record the name of the survey leader
                "SurveyLeaderName": write_SurveyLeaderName(),
                # Select the name of the linked Marine Reserve
                "LinkToMarineReserve": select_LinkToMarineReserve(
                    choices_df.MarineReserve.dropna().unique().tolist()
                ),
                # Specify if survey is single species
                "FishMultiSpecies": select_FishMultiSpecies(),
                # Specify how the survey was stratified
                "StratifiedBy": select_StratifiedBy(
                    choices_df.StratifiedBy.dropna().unique().tolist()
                ),
                # Select if survey is part of long term monitoring
                "IsLongTermMonitoring": select_IsLongTermMonitoring(),
                # Specify the site selection of the survey
                "SiteSelectionDesign": select_SiteSelectionDesign(
                    choices_df.SiteSelection.dropna().unique().tolist()
                ),
                # Specify the unit selection of the survey
                "UnitSelectionDesign": select_UnitSelectionDesign(
                    choices_df.UnitSelection.dropna().unique().tolist()
                ),
                # Select the type of right holder of the survey
                "RightsHolder": select_RightsHolder(
                    choices_df.RightsHolder.dropna().unique().tolist()
                ),
                # Write who can access the videos/resources
                "AccessRights": select_AccessRights(),
                # Write a description of the survey design and objectives
                "SurveyVerbatim": write_SurveyVerbatim(),
                # Select the type of BUV
                "BUVType": select_BUVType(
                    choices_df.BUVType.dropna().unique().tolist()
                ),
                # Write the link to the pictures
                "LinkToPicture": write_LinkToPicture(),
                # Write the name of the vessel
                "Vessel": write_Vessel(),
                # Write the link to the fieldsheets
                "LinkToFieldSheets": write_LinkToFieldSheets(),
                # Write the link to LinkReport01
                "LinkReport01": write_LinkReport01(),
                # Write the link to LinkReport02
                "LinkReport02": write_LinkReport02(),
                # Write the link to LinkReport03
                "LinkReport03": write_LinkReport03(),
                # Write the link to LinkReport04
                "LinkReport04": write_LinkReport04(),
                # Write the link to LinkToOriginalData
                "LinkToOriginalData": write_LinkToOriginalData(),
            }

            return survey_info

    w = interactive(
        f,
        Existing_or_new=widgets.Dropdown(
            options=["Existing", "New survey"],
            description="Existing or new survey:",
            disabled=False,
            layout=Layout(width="90%"),
            style={"description_width": "initial"},
        ),
    )

    display(w)
    return w


def record_encoder():
    """
    > This function creates a widget that asks for the name of the person encoding the survey
    information
    :return: The name of the person encoding the survey information
    """
    # Widget to record the encoder of the survey information
    EncoderName_widget = widgets.Text(
        placeholder="First and last name",
        description="Name of the person encoding this survey information:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(EncoderName_widget)
    return EncoderName_widget


def select_SurveyStartDate():
    """
    > This function creates a widget that allows the user to select a date
    :return: A widget that allows the user to select a date.
    """
    # Widget to record the start date of the survey
    SurveyStartDate_widget = widgets.DatePicker(
        description="Offical date when survey started as a research event",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(SurveyStartDate_widget)
    return SurveyStartDate_widget


def write_SurveyName():
    """
    > This function creates a widget that allows the user to enter a name for the survey
    :return: A widget object
    """
    # Widget to record the name of the survey
    SurveyName_widget = widgets.Text(
        placeholder="Baited Underwater Video Taputeranga Apr 2015",
        description="A name for this survey:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(SurveyName_widget)
    return SurveyName_widget


def select_OfficeName(OfficeName_options: list):
    """
    > This function creates a dropdown widget that allows the user to select the name of the DOC Office
    responsible for the survey

    :param OfficeName_options: a list of the names of the DOC offices that are responsible for the survey
    :return: The widget is being returned.
    """
    # Widget to record the name of the linked DOC Office
    OfficeName_widget = widgets.Dropdown(
        options=OfficeName_options,
        description="Department of Conservation Office responsible for this survey:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(OfficeName_widget)
    return OfficeName_widget


def write_ContractorName():
    """
    > This function creates a widget that allows the user to enter the name of the contractor
    :return: The widget is being returned.
    """
    # Widget to record the name of the contractor
    ContractorName_widget = widgets.Text(
        placeholder="No contractor",
        description="Person/company contracted to carry out the survey:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(ContractorName_widget)
    return ContractorName_widget


def write_ContractNumber():
    """
    > This function creates a widget that allows the user to enter the contract number for the survey
    :return: The widget is being returned.
    """
    # Widget to record the number of the contractor
    ContractNumber_widget = widgets.Text(
        description="Contract number for this survey:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(ContractNumber_widget)
    return ContractNumber_widget


def write_LinkToContract():
    """
    > This function creates a widget that allows the user to enter a hyperlink to the contract related
    to the survey
    :return: The widget
    """
    # Widget to record the link to the contract
    LinkToContract_widget = widgets.Text(
        description="Hyperlink to the DOCCM for the contract related to this survey:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(LinkToContract_widget)
    return LinkToContract_widget


def write_SurveyLeaderName():
    """
    > This function creates a widget that allows the user to enter the name of the person in charge of
    the survey
    :return: The widget is being returned.
    """
    # Widget to record the name of the survey leader
    SurveyLeaderName_widget = widgets.Text(
        placeholder="First and last name",
        description="Name of the person in charge of this survey:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(SurveyLeaderName_widget)
    return SurveyLeaderName_widget


def select_LinkToMarineReserve(reserves_available: list):
    """
    > This function creates a dropdown widget that allows the user to select the name of the Marine
    Reserve that the survey is linked to

    :param reserves_available: a list of the names of the Marine Reserves that are available to be linked to the survey
    :return: The name of the Marine Reserve linked to the survey
    """
    # Widget to record the name of the linked Marine Reserve
    LinkToMarineReserve_widget = widgets.Dropdown(
        options=reserves_available,
        description="Marine Reserve linked to the survey:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(LinkToMarineReserve_widget)
    return LinkToMarineReserve_widget


def select_FishMultiSpecies():
    """
    > This function creates a widget that allows the user to select whether the survey is a single
    species survey or not
    :return: A widget that can be used to select a single species or multiple species
    """

    # Widget to record if survey is single species
    def FishMultiSpecies_to_true_false(FishMultiSpecies_value):
        if FishMultiSpecies_value == "Yes":
            return False
        else:
            return True

    w = interactive(
        FishMultiSpecies_to_true_false,
        FishMultiSpecies_value=widgets.Dropdown(
            options=["No", "Yes"],
            description="Does this survey look at a single species?",
            disabled=False,
            layout=Layout(width="95%"),
            style={"description_width": "initial"},
        ),
    )
    display(w)
    return w


def select_StratifiedBy(StratifiedBy_choices: list):
    """
    > This function creates a dropdown widget that allows the user to select the stratified factors for
    the sampling design

    :param StratifiedBy_choices: a list of choices for the dropdown menu
    :return: A widget object
    """
    # Widget to record if survey was stratified by any factor
    StratifiedBy_widget = widgets.Dropdown(
        options=StratifiedBy_choices,
        description="Stratified factors for the sampling design",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(StratifiedBy_widget)
    return StratifiedBy_widget


def select_IsLongTermMonitoring():
    """
    > This function creates a widget that allows the user to select whether the survey is part of a
    long-term monitoring
    :return: A widget object
    """

    # Widget to record if survey is part of long term monitoring
    def IsLongTermMonitoring_to_true_false(IsLongTermMonitoring_value):
        if IsLongTermMonitoring_value == "No":
            return False
        else:
            return True

    w = interactive(
        IsLongTermMonitoring_to_true_false,
        IsLongTermMonitoring_value=widgets.Dropdown(
            options=["Yes", "No"],
            description="Is the survey part of a long-term monitoring?",
            disabled=False,
            layout=Layout(width="95%"),
            style={"description_width": "initial"},
        ),
    )
    display(w)
    return w


def select_SiteSelectionDesign(site_selection_options: list):
    """
    This function creates a dropdown widget that allows the user to select the site selection design of
    the survey

    :param site_selection_options: a list of strings that are the options for the dropdown menu
    :return: A widget
    """
    # Widget to record the site selection of the survey
    SiteSelectionDesign_widget = widgets.Dropdown(
        options=site_selection_options,
        description="What was the design for site selection?",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(SiteSelectionDesign_widget)
    return SiteSelectionDesign_widget


def select_UnitSelectionDesign(unit_selection_options: list):
    """
    > This function creates a dropdown widget that allows the user to select the design for site
    selection

    :param unit_selection_options: list
    :type unit_selection_options: list
    :return: A widget that allows the user to select the unit selection design of the survey.
    """
    # Widget to record the unit selection of the survey
    UnitSelectionDesign_widget = widgets.Dropdown(
        options=unit_selection_options,
        description="What was the design for unit selection?",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(UnitSelectionDesign_widget)
    return UnitSelectionDesign_widget


def select_RightsHolder(RightsHolder_options: list):
    """
    > This function creates a dropdown widget that allows the user to select the type of right holder of
    the survey

    :param RightsHolder_options: a list of options for the dropdown menu
    :return: A widget
    """
    # Widget to record the type of right holder of the survey
    RightsHolder_widget = widgets.Dropdown(
        options=RightsHolder_options,
        description="Person(s) or organization(s) owning or managing rights over the resource",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(RightsHolder_widget)
    return RightsHolder_widget


def select_AccessRights():
    """
    > This function creates a widget that asks the user to enter information about who can access the
    resource
    :return: A widget object
    """
    # Widget to record information about who can access the resource
    AccessRights_widget = widgets.Text(
        placeholder="",
        description="Who can access the resource?",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(AccessRights_widget)
    return AccessRights_widget


def write_SurveyVerbatim():
    """
    > This function creates a widget that allows the user to enter a description of the survey design
    and objectives
    :return: A widget
    """
    # Widget to record description of the survey design and objectives
    SurveyVerbatim_widget = widgets.Textarea(
        placeholder="",
        description="Provide an exhaustive description of the survey design and objectives",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(SurveyVerbatim_widget)
    return SurveyVerbatim_widget


def select_BUVType(BUVType_choices: list):
    """
    > This function creates a dropdown widget that allows the user to select the type of BUV used for
    the survey

    :param BUVType_choices: list
    :type BUVType_choices: list
    :return: A widget
    """
    # Widget to record the type of BUV
    BUVType_widget = widgets.Dropdown(
        options=BUVType_choices,
        description="Type of BUV used for the survey:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(BUVType_widget)
    return BUVType_widget


def write_LinkToPicture():
    """
    > This function creates a text box for the user to enter the link to the DOCCM folder for the survey
    photos
    :return: The widget is being returned.
    """
    # Widget to record the link to the pictures
    LinkToPicture_widget = widgets.Text(
        description="Hyperlink to the DOCCM folder for this survey photos:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(LinkToPicture_widget)
    return LinkToPicture_widget


def write_Vessel():
    """
    This function creates a widget that allows the user to enter the name of the vessel that deployed
    the unit
    :return: The name of the vessel used to deploy the unit.
    """
    # Widget to record the name of the vessel
    Vessel_widget = widgets.Text(
        description="Vessel used to deploy the unit:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(Vessel_widget)
    return Vessel_widget


def write_LinkToFieldSheets():
    """
    **write_LinkToFieldSheets()**: This function creates a text box for the user to enter a hyperlink to
    the DOCCM for the field sheets used to gather the survey information
    :return: The text box widget.
    """
    LinkToFieldSheets = widgets.Text(
        description="Hyperlink to the DOCCM for the field sheets used to gather the survey information:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(LinkToFieldSheets)
    return LinkToFieldSheets


def write_LinkReport01():
    """
    > This function creates a text box for the user to enter a hyperlink to the first (of up to four)
    DOCCM report related to these data
    :return: The text box.
    """
    LinkReport01 = widgets.Text(
        description="Hyperlink to the first (of up to four) DOCCM report related to these data:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(LinkReport01)
    return LinkReport01


def write_LinkReport02():
    """
    **write_LinkReport02()**: This function creates a text box for the user to enter a hyperlink to the
    second DOCCM report related to these data
    :return: The text box widget.
    """
    LinkReport02 = widgets.Text(
        description="Hyperlink to the second DOCCM report related to these data:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(LinkReport02)
    return LinkReport02


def write_LinkReport03():
    """
    **write_LinkReport03()**: This function creates a text box for the user to enter a hyperlink to the
    third DOCCM report related to these data
    :return: The text box widget.
    """
    LinkReport03 = widgets.Text(
        description="Hyperlink to the third DOCCM report related to these data:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(LinkReport03)
    return LinkReport03


def write_LinkReport04():
    """
    **write_LinkReport04()**: This function creates a text box for the user to enter a hyperlink to the
    fourth DOCCM report related to these data
    :return: The text box widget.
    """
    LinkReport04 = widgets.Text(
        description="Hyperlink to the fourth DOCCM report related to these data:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(LinkReport04)
    return LinkReport04


def write_LinkToOriginalData():
    """
    > This function creates a text box that allows the user to enter a hyperlink to the DOCCM for the
    spreadsheet where these data were intially encoded
    :return: A text box with a description.
    """
    LinkToOriginalData = widgets.Text(
        description="Hyperlink to the DOCCM for the spreadsheet where these data were intially encoded:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(LinkToOriginalData)
    return LinkToOriginalData


# Confirm the details of the survey
def confirm_survey(survey_i, db_info_dict: dict):
    """
    It takes the survey information and checks if it's a new survey or an existing one. If it's a new
    survey, it saves the information in the survey csv file. If it's an existing survey, it prints the
    information for the pre-existing survey

    :param survey_i: the survey widget
    :param db_info_dict: a dictionary with the following keys:
    """

    correct_button = widgets.Button(
        description="Yes, details are correct",
        layout=Layout(width="25%"),
        style={"description_width": "initial"},
        button_style="danger",
    )

    wrong_button = widgets.Button(
        description="No, I will go back and fix them",
        layout=Layout(width="45%"),
        style={"description_width": "initial"},
        button_style="danger",
    )

    # If new survey, review details and save changes in survey csv server
    if isinstance(survey_i.result, dict):
        # Save the responses as a new row for the survey csv file
        new_survey_row_dict = {
            key: (
                value.value
                if hasattr(value, "value")
                else value.result
                if isinstance(value.result, int)
                else value.result.value
            )
            for key, value in survey_i.result.items()
        }
        new_survey_row = pd.DataFrame.from_records(new_survey_row_dict, index=[0])

        # Load the csv with sites and survey choices
        choices_df = pd.read_csv(db_info_dict["local_choices_csv"])

        # Get prepopulated fields for the survey
        new_survey_row["OfficeContact"] = choices_df[
            choices_df["OfficeName"] == new_survey_row.OfficeName.values[0]
        ]["OfficeContact"].values[0]
        new_survey_row[["SurveyLocation", "Region"]] = choices_df[
            choices_df["MarineReserve"] == new_survey_row.LinkToMarineReserve.values[0]
        ][["MarineReserveAbreviation", "Region"]].values[0]
        new_survey_row["DateEntry"] = datetime.date.today()
        new_survey_row["SurveyType"] = "BUV"
        new_survey_row["SurveyID"] = (
            new_survey_row["SurveyLocation"]
            + "_"
            + new_survey_row["SurveyStartDate"].values[0].strftime("%Y%m%d")
            + "_"
            + new_survey_row["SurveyType"]
        )

        # Review details
        print("The details of the new survey are:")
        for ind in new_survey_row.T.index:
            print(ind, "-->", new_survey_row.T[0][ind])

        # Save changes in survey csv locally and in the server
        async def f(new_survey_row):
            x = await t_utils.wait_for_change(
                correct_button, wrong_button
            )  # <---- Pass both buttons into the function
            if (
                x == "Yes, details are correct"
            ):  # <--- use if statement to trigger different events for the two buttons
                # Load the csv with sites information
                surveys_df = pd.read_csv(db_info_dict["local_surveys_csv"])

                # Drop unnecessary columns
                #                 new_survey_row = new_survey_row.drop(columns=['ShortFolder'])

                # Check the columns are the same
                diff_columns = list(
                    set(surveys_df.columns.sort_values().values)
                    - set(new_survey_row.columns.sort_values().values)
                )

                if len(diff_columns) > 0:
                    logging.error(
                        f"The {diff_columns} columns are missing from the survey information."
                    )
                    raise

                # Check if the survey exist in the csv
                if new_survey_row.SurveyID.unique()[0] in surveys_df.SurveyID.unique():
                    logging.error(
                        f"The survey {new_survey_row.SurveyID.unique()[0]} already exists in the database."
                    )
                    raise

                print("Updating the new survey information.")

                # Add the new row to the choices df
                surveys_df = surveys_df.append(new_survey_row, ignore_index=True)

                # Save the updated df locally
                surveys_df.to_csv(db_info_dict["local_surveys_csv"], index=False)

                # Save the updated df in the server
                server_utils.upload_file_to_s3(
                    db_info_dict["client"],
                    bucket=db_info_dict["bucket"],
                    key=db_info_dict["server_surveys_csv"],
                    filename=db_info_dict["local_surveys_csv"].__str__(),
                )

                print("Survey information updated!")

            else:
                print("Come back when the data is tidy!")

    # If existing survey print the info for the pre-existing survey
    else:
        # Load the csv with surveys information
        surveys_df = pd.read_csv(db_info_dict["local_surveys_csv"])

        # Select the specific survey info
        new_survey_row = surveys_df[
            surveys_df["SurveyName"] == survey_i.result.value
        ].reset_index(drop=True)

        print("The details of the selected survey are:")
        for ind in new_survey_row.T.index:
            print(ind, "-->", new_survey_row.T[0][ind])

        async def f(new_survey_row):
            x = await t_utils.wait_for_change(
                correct_button, wrong_button
            )  # <---- Pass both buttons into the function
            if (
                x == "Yes, details are correct"
            ):  # <--- use if statement to trigger different events for the two buttons
                print("Great, you can start uploading the movies.")

            else:
                print("Come back when the data is tidy!")

    print("")
    print("")
    print("Are the survey details above correct?")
    display(
        HBox([correct_button, wrong_button])
    )  # <----Display both buttons in an HBox
    asyncio.create_task(f(new_survey_row))


####################################################
############### MOVIES FUNCTIONS ###################
####################################################


def get_survey_name(survey_i):
    """
    If the survey is new, save the responses for the new survey as a dataframe. If the survey is
    existing, return the name of the survey

    :param survey_i: the survey object
    :return: The name of the survey
    """
    # If new survey, review details and save changes in survey csv server
    if isinstance(survey_i.result, dict):
        # Save the responses for the new survey as a dataframe
        new_survey_row_dict = {
            key: (
                value.value
                if hasattr(value, "value")
                else value.result
                if isinstance(value.result, int)
                else value.result.value
            )
            for key, value in survey_i.result.items()
        }
        survey_name = new_survey_row_dict["SurveyName"]

    # If existing survey print the info for the pre-existing survey
    else:
        # Return the name of the survey
        survey_name = survey_i.result.value

    return survey_name


def select_deployment(project, db_info_dict: dict, survey_i):
    """
    This function allows the user to select a deployment from a survey of interest

    :param project: the name of the project
    :param db_info_dict: a dictionary with the following keys:
    :type db_info_dict: dict
    :param survey_i: the index of the survey you want to download
    """
    # Load the csv with with sites and survey choices
    choices_df = pd.read_csv(db_info_dict["local_choices_csv"])

    # Read surveys csv
    surveys_df = pd.read_csv(
        db_info_dict["local_surveys_csv"], parse_dates=["SurveyStartDate"]
    )

    # Get the name of the survey
    survey_name = get_survey_name(survey_i)

    # Save the SurveyID that match the survey name
    survey_row = surveys_df[surveys_df["SurveyName"] == survey_name].reset_index(
        drop=True
    )

    # Save the year of the survey
    #     survey_year = survey_row["SurveyStartDate"].values[0].strftime("%Y")
    survey_year = survey_row["SurveyStartDate"].dt.year.values[0]

    # Get prepopulated fields for the survey
    survey_row[["ShortFolder"]] = choices_df[
        choices_df["MarineReserve"] == survey_row.LinkToMarineReserve.values[0]
    ][["ShortFolder"]].values[0]

    # Save the "server filename" of the survey
    short_marine_reserve = survey_row["ShortFolder"].values[0]

    # Save the "server filename" of the survey
    survey_server_name = short_marine_reserve + "-buv-" + str(survey_year) + "/"

    # Retrieve deployments info from the survey of interest
    deployments_server_name = db_info_dict["client"].list_objects(
        Bucket=db_info_dict["bucket"], Prefix=survey_server_name, Delimiter="/"
    )
    # Convert info to dataframe
    deployments_server_name = pd.DataFrame.from_dict(
        deployments_server_name["CommonPrefixes"]
    )

    # Select only the name of the "deployment folder"
    deployments_server_name_list = deployments_server_name.Prefix.str.split("/").str[1]

    # Widget to select the deployment of interest
    deployment_widget = widgets.SelectMultiple(
        options=deployments_server_name_list,
        description="New deployment:",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(deployment_widget)
    return deployment_widget, survey_row, survey_server_name


def select_eventdate(survey_row: pd.DataFrame):
    """
    > This function creates a date picker widget that allows the user to select the date of the survey.
    The default date is the beginning of the survey

    :param survey_row: a dataframe containing survey information
    :return: A widget object
    """
    # Set the beginning of the survey as default date
    default_date = pd.Timestamp(survey_row["SurveyStartDate"].values[0]).to_pydatetime()

    # Select the date
    date_widget = widgets.DatePicker(
        description="Date of deployment:",
        value=default_date,
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(date_widget)
    return date_widget


def check_deployment(
    deployment_selected: widgets.Widget,
    deployment_date: widgets.Widget,
    survey_server_name: str,
    db_info_dict: dict,
    survey_i,
):
    """
    This function checks if the deployment selected by the user is already in the database. If it is, it
    will raise an error. If it is not, it will return the deployment filenames

    :param deployment_selected: a list of the deployment names selected by the user
    :param deployment_date: the date of the deployment
    :param survey_server_name: The name of the survey in the server
    :param db_info_dict: a dictionary containing the following keys:
    :param survey_i: the index of the survey you want to upload to
    :return: A list of deployment filenames
    """
    # Ensure at least one deployment has been selected
    if not deployment_selected.value:
        logging.error("Please select a deployment.")
        raise

    # Get list of movies available in server from that survey
    deployments_in_server_df = server_utils.get_matching_s3_keys(
        db_info_dict["client"], db_info_dict["bucket"], prefix=survey_server_name
    )

    # Get a list of the movies in the server
    files_in_server = deployments_in_server_df.Key.str.split("/").str[-1].to_list()
    deployments_in_server = [
        file
        for file in files_in_server
        if file[-3:] in movie_utils.get_movie_extensions()
    ]

    # Read surveys csv
    surveys_df = pd.read_csv(
        db_info_dict["local_surveys_csv"], parse_dates=["SurveyStartDate"]
    )

    # Get the name of the survey
    survey_name = get_survey_name(survey_i)

    # Save the SurveyID that match the survey name
    SurveyID = surveys_df[surveys_df["SurveyName"] == survey_name].SurveyID.to_list()[0]

    # Read movie.csv info
    movies_df = pd.read_csv(db_info_dict["local_movies_csv"])

    # Get a list of deployment names from the csv of the survey of interest
    deployment_in_csv = movies_df[movies_df["SurveyID"] == SurveyID].filename.to_list()

    # Save eventdate as str
    EventDate_str = deployment_date.value.strftime("%d_%m_%Y")

    deployment_filenames = []
    for deployment_i in deployment_selected.value:
        # Specify the name of the deployment
        deployment_name = deployment_i + "_" + EventDate_str

        if deployment_name in deployment_in_csv:
            logging.error(
                f"Deployment {deployment_name} already exist in the csv file reselect new deployments before going any further!"
            )
            raise

        # Specify the filename of the deployment
        filename = deployment_name + ".MP4"

        if filename in deployments_in_server:
            logging.error(
                f"Deployment {deployment_name} already exist in the server reselect new deployments before going any further!"
            )
            raise

        else:
            deployment_filenames = deployment_filenames + [filename]
    print(
        "There is no existing information in the database for",
        deployment_filenames,
        "You can continue uploading the information.",
    )

    return deployment_filenames


def update_new_deployments(
    deployment_filenames: list,
    db_info_dict: dict,
    survey_server_name: str,
    deployment_date: widgets.Widget,
):
    """
    It takes a list of filenames, a dictionary with the database information, the name of the server,
    and the date of the deployment, and it returns a list of the movies in the server

    :param deployment_filenames: a list of the filenames of the videos you want to concatenate
    :param db_info_dict: a dictionary with the following keys:
    :param survey_server_name: the name of the folder in the server where the survey is stored
    :param deployment_date: the date of the deployment
    :return: A list of the movies in the server
    """
    for deployment_i in deployment_filenames:
        # Save eventdate as str
        EventDate_str = deployment_date.value.strftime("%d_%m_%Y")

        # Get the site information from the filename
        site_deployment = deployment_i.replace("_" + EventDate_str + ".MP4", "")
        print(site_deployment)

        # Specify the folder with files in the server
        deployment_folder = survey_server_name + site_deployment

        # Retrieve files info from the deployment folder of interest
        deployments_files = server_utils.get_matching_s3_keys(
            db_info_dict["client"], db_info_dict["bucket"], prefix=deployment_folder
        )
        # Get a list of the movies in the server
        files_server = deployments_files.Key.to_list()
        movie_files_server = [
            file
            for file in files_server
            if file[-3:] in movie_utils.get_movie_extensions()
        ]

        # Concatenate the files if multiple
        if len(movie_files_server) > 1:
            print("The files", movie_files_server, "will be concatenated")

            # Save filepaths in a text file
            textfile = open("a_file.txt", "w")

            for movie_i in sorted(movie_files_server):
                temp_i = movie_i.split("/")[2]
                server_utils.download_object_from_s3(
                    client=db_info_dict["client"],
                    bucket=db_info_dict["bucket"],
                    key=movie_i,
                    filename=temp_i,
                )

                textfile.write("file '" + temp_i + "'" + "\n")
            textfile.close()

            if not os.path.exists(deployment_i):
                print("Concatenating ", deployment_i)

                # Concatenate the videos
                subprocess.call(
                    [
                        "ffmpeg",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        "a_file.txt",
                        "-c",
                        "copy",
                        deployment_i,
                    ]
                )

                print(deployment_i, "concatenated successfully")

        # Change the filename if only one file
        else:
            print("WIP")

        return movie_files_server


def record_deployment_info(deployment_filenames: list, db_info_dict: dict):
    """
    This function takes in a list of deployment filenames and a dictionary of database information, and
    returns a dictionary of deployment information.

    :param deployment_filenames: a list of the filenames of the movies you want to add to the database
    :param db_info_dict: a dictionary with the following keys:
    :return: A dictionary with the deployment info
    """

    for deployment_i in deployment_filenames:
        # Estimate the fps and length info
        fps, duration = movie_utils.get_length(deployment_i, os.getcwd())

        # Read csv as pd
        movies_df = pd.read_csv(db_info_dict["local_movies_csv"])

        # Load the csv with with sites and survey choices
        choices_df = pd.read_csv(db_info_dict["local_choices_csv"])

        deployment_info = {
            # Select the start of the sampling
            "SamplingStart": select_SamplingStart(duration),
            # Select the end of the sampling
            "SamplingEnd": select_SamplingEnd(duration),
            # Specify if deployment is bad
            "IsBadDeployment": select_IsBadDeployment(),
            # Write the number of the replicate within the site
            "ReplicateWithinSite": write_ReplicateWithinSite(),
            # Select the person who recorded this deployment
            "RecordedBy": select_RecordedBy(movies_df.RecordedBy.unique()),
            # Select depth stratum of the deployment
            "DepthStrata": select_DepthStrata(),
            # Select the depth of the deployment
            "Depth": select_Depth(),
            # Select the underwater visibility
            "UnderwaterVisibility": select_UnderwaterVisibility(
                choices_df.UnderwaterVisibility.dropna().unique().tolist()
            ),
            # Select the time in
            "TimeIn": deployment_TimeIn(),
            # Select the time out
            "TimeOut": deployment_TimeOut(),
            # Add any comment related to the deployment
            "NotesDeployment": write_NotesDeployment(),
            # Select the theoretical duration of the deployment
            "DeploymentDurationMinutes": select_DeploymentDurationMinutes(),
            # Write the type of habitat
            "Habitat": write_Habitat(),
            # Write the number of NZMHCS_Abiotic
            "NZMHCS_Abiotic": write_NZMHCS_Abiotic(),
            # Write the number of NZMHCS_Biotic
            "NZMHCS_Biotic": write_NZMHCS_Biotic(),
            # Select the level of the tide
            "TideLevel": select_TideLevel(
                choices_df.TideLevel.dropna().unique().tolist()
            ),
            # Describe the weather of the deployment
            "Weather": write_Weather(),
            # Select the model of the camera used
            "CameraModel": select_CameraModel(
                choices_df.CameraModel.dropna().unique().tolist()
            ),
            # Write the camera lens and settings used
            "LensModel": write_LensModel(),
            # Specify the type of bait used
            "BaitSpecies": write_BaitSpecies(),
            # Specify the amount of bait used
            "BaitAmount": select_BaitAmount(),
        }

        return deployment_info


def select_SamplingStart(duration_i: int):
    # Select the start of the survey
    surv_start = interactive(
        to_hhmmss,
        seconds=widgets.IntSlider(
            value=0,
            min=0,
            max=duration_i,
            step=1,
            description="Survey starts (seconds):",
            layout=Layout(width="50%"),
            style={"description_width": "initial"},
        ),
    )
    display(surv_start)
    return surv_start


def select_SamplingEnd(duration_i: int):
    #     # Set default to 30 mins or max duration
    #     start_plus_30 = surv_start_i+(30*60)

    #     if start_plus_30>duration_i:
    #         default_end = duration_i
    #     else:
    #         default_end = start_plus_30

    # Select the end of the survey
    surv_end = interactive(
        to_hhmmss,
        seconds=widgets.IntSlider(
            value=duration_i,
            min=0,
            max=duration_i,
            step=1,
            description="Survey ends (seconds):",
            layout=Layout(width="50%"),
            style={"description_width": "initial"},
        ),
    )
    display(surv_end)
    return surv_end


def select_IsBadDeployment():
    """
    > This function creates a dropdown widget that allows the user to select whether or not the
    deployment is bad
    :return: A widget object
    """

    def deployment_to_true_false(deploy_value):
        if deploy_value == "No, it is a great video":
            return False
        else:
            return True

    w = interactive(
        deployment_to_true_false,
        deploy_value=widgets.Dropdown(
            options=["Yes, unfortunately it is marine crap", "No, it is a great video"],
            value="No, it is a great video",
            description="Is it a bad deployment?",
            disabled=False,
            layout=Layout(width="50%"),
            style={"description_width": "initial"},
        ),
    )
    display(w)
    return w


def write_ReplicateWithinSite():
    """
    This function creates a widget that allows the user to select the depth of the deployment
    :return: The value of the widget.
    """
    # Select the depth of the deployment
    ReplicateWithinSite_widget = widgets.BoundedIntText(
        value=0,
        min=0,
        max=1000,
        step=1,
        description="Number of the replicate within site (Field number of planned BUV station):",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(ReplicateWithinSite_widget)
    return ReplicateWithinSite_widget


# Select the person who recorded the deployment
def select_RecordedBy(existing_recorders: list):
    """
    This function takes a list of existing recorders and returns a widget that allows the user to select
    an existing recorder or enter a new one

    :param existing_recorders: a list of existing recorders
    :return: A widget with a dropdown menu that allows the user to select between two options, 'Existing' and 'New author'.
    """

    def f(Existing_or_new):
        if Existing_or_new == "Existing":
            RecordedBy_widget = widgets.Dropdown(
                options=existing_recorders,
                description="Existing recorder:",
                disabled=False,
                layout=Layout(width="50%"),
                style={"description_width": "initial"},
            )

        if Existing_or_new == "New author":
            RecordedBy_widget = widgets.Text(
                placeholder="First and last name",
                description="Recorder:",
                disabled=False,
                layout=Layout(width="50%"),
                style={"description_width": "initial"},
            )

        display(RecordedBy_widget)
        return RecordedBy_widget

    w = interactive(
        f,
        Existing_or_new=widgets.Dropdown(
            options=["Existing", "New author"],
            description="Deployment recorded by existing or new person:",
            disabled=False,
            layout=Layout(width="50%"),
            style={"description_width": "initial"},
        ),
    )
    display(w)
    return w


def select_DepthStrata():
    # Select the depth of the deployment
    deployment_DepthStrata = widgets.Text(
        placeholder="5-25m",
        description="Depth stratum within which the BUV unit was deployed:",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(deployment_DepthStrata)
    return deployment_DepthStrata


def select_Depth():
    # Select the depth of the deployment
    deployment_depth = widgets.BoundedIntText(
        value=0,
        min=0,
        max=100,
        step=1,
        description="Depth reading in meters at the time of BUV unit deployment:",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(deployment_depth)
    return deployment_depth


def select_UnderwaterVisibility(visibility_options: list):
    """
    > This function creates a dropdown menu with the options of the visibility of the water during the
    video deployment

    :param visibility_options: a list of options for the dropdown menu
    :return: The dropdown menu with the options for the water visibility of the video deployment.
    """
    UnderwaterVisibility = widgets.Dropdown(
        options=visibility_options,
        description="Water visibility of the video deployment:",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(UnderwaterVisibility)
    return UnderwaterVisibility


def deployment_TimeIn():
    # Select the TimeIn
    TimeIn_widget = widgets.TimePicker(
        description="Time in of the deployment:",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(TimeIn_widget)
    return TimeIn_widget


def deployment_TimeOut():
    # Select the TimeOut
    TimeOut_widget = widgets.TimePicker(
        description="Time out of the deployment:",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(TimeOut_widget)
    return TimeOut_widget


# Write a comment about the deployment
def write_NotesDeployment():
    # Create the comment widget
    comment_widget = widgets.Text(
        placeholder="Type comment",
        description="Comment:",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(comment_widget)
    return comment_widget


def select_DeploymentDurationMinutes():
    # Select the theoretical duration of the deployment
    DeploymentDurationMinutes = widgets.BoundedIntText(
        value=0,
        min=0,
        max=60,
        step=1,
        description="Theoretical minimum soaking time for the unit (mins):",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(DeploymentDurationMinutes)
    return DeploymentDurationMinutes


def write_Habitat():
    # Widget to record the type of habitat
    Habitat_widget = widgets.Text(
        placeholder="Make and model",
        description="Describe the nature of the seabed (mud, sand, gravel, cobbles, rocky reef, kelp forest…)",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(Habitat_widget)
    return Habitat_widget


def write_NZMHCS_Abiotic():
    # Widget to record the type of NZMHCS_Abiotic
    NZMHCS_Abiotic_widget = widgets.Text(
        placeholder="0001",
        description="Write the Abiotic New Zealand Marine Habitat Classification number (Table 5)",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(NZMHCS_Abiotic_widget)
    return NZMHCS_Abiotic_widget


def write_NZMHCS_Biotic():
    # Widget to record the type of NZMHCS_Biotic
    NZMHCS_Biotic_widget = widgets.Text(
        placeholder="0001",
        description="Write the Biotic New Zealand Marine Habitat Classification number (Table 6)",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(NZMHCS_Biotic_widget)
    return NZMHCS_Biotic_widget


# Widget to record the level of the tide
def select_TideLevel(TideLevel_choices: list):
    TideLevel_widget = widgets.Dropdown(
        options=TideLevel_choices,
        description="Tidal level at the time of sampling:",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(TideLevel_widget)
    return TideLevel_widget


# Widget to record the weather
def write_Weather():
    Weather_widget = widgets.Text(
        description="Describe the weather for the survey:",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(Weather_widget)
    return Weather_widget


def select_CameraModel(CameraModel_choices: list):
    # Widget to record the type of camera
    CameraModel_widget = widgets.Dropdown(
        options=CameraModel_choices,
        description="Select the make and model of camera used",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(CameraModel_widget)
    return CameraModel_widget


def write_LensModel():
    # Widget to record the camera settings
    LensModel_widget = widgets.Text(
        placeholder="Wide lens, 1080x1440",
        description="Describe the camera lens and settings",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(LensModel_widget)
    return LensModel_widget


def write_BaitSpecies():
    # Widget to record the type of bait used
    BaitSpecies_widget = widgets.Text(
        placeholder="Pilchard",
        description="Species that was used as bait for the deployment",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(BaitSpecies_widget)
    return BaitSpecies_widget


def select_BaitAmount():
    # Widget to record the amount of bait used
    BaitAmount_widget = widgets.BoundedIntText(
        value=500,
        min=100,
        max=1000,
        step=1,
        description="Amount of bait used (g):",
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )
    display(BaitAmount_widget)
    return BaitAmount_widget


# Display in hours, minutes and seconds
def to_hhmmss(seconds):
    print("Time selected:", datetime.timedelta(seconds=seconds))
    return seconds


def confirm_deployment_details(
    deployment_names: list,
    survey_server_name: str,
    db_info_dict: dict,
    survey_i,
    deployment_info: list,
    movie_files_server: str,
    deployment_date: widgets.Widget,
):
    """
    This function takes the deployment information and the survey information and returns a dataframe
    with the deployment information

    :param deployment_names: list of deployment names
    :param survey_server_name: The name of the folder in the server where the survey is located
    :param db_info_dict: a dictionary with the following keys:
    :param survey_i: the survey name
    :param deployment_info: a list of dictionaries with the deployment information
    :param movie_files_server: the path to the folder with the movie files in the server
    :param deployment_date: The date of the deployment
    :return: The new_movie_row is a dataframe with the information of the new deployment.
    """

    for deployment_i in deployment_names:
        # Save the deployment responses as a new row for the movies csv file
        new_movie_row_dict = {
            key: (
                value.value
                if hasattr(value, "value")
                else value.result
                if isinstance(value.result, int)
                else value.result.value
            )
            for key, value in deployment_info[
                deployment_names.index(deployment_i)
            ].items()
        }

        new_movie_row = pd.DataFrame.from_records(new_movie_row_dict, index=[0])

        # Read movies csv
        movies_df = pd.read_csv(db_info_dict["local_movies_csv"])

        # Get prepopulated fields for the movie deployment
        # Add movie id
        new_movie_row["movie_id"] = 1 + movies_df.movie_id.iloc[-1]

        # Read surveys csv
        surveys_df = pd.read_csv(db_info_dict["local_surveys_csv"])

        # Save the name of the survey
        if isinstance(survey_i.result, dict):
            # Load the ncsv with survey information
            surveys_df = pd.read_csv(db_info_dict["local_surveys_csv"])

            # Save the SurveyID of the last survey added
            new_movie_row["SurveyID"] = surveys_df.tail(1)["SurveyID"].values[0]

            # Save the name of the survey
            survey_name = surveys_df.tail(1)["SurveyName"].values[0]

        else:
            # Return the name of the survey
            survey_name = survey_i.result.value

            # Save the SurveyID that match the survey name
            new_movie_row["SurveyID"] = surveys_df[
                surveys_df["SurveyName"] == survey_name
            ]["SurveyID"].values[0]

        # Get the site information from the filename
        site_deployment = "_".join(deployment_i.split("_")[:2])

        # Specify the folder with files in the server
        deployment_folder = survey_server_name + site_deployment

        # Create temporary prefix (s3 path) for concatenated video
        new_movie_row["prefix_conc"] = deployment_folder + "/" + deployment_i

        # Select previously processed movies within the same survey
        survey_movies_df = movies_df[
            movies_df["SurveyID"] == new_movie_row["SurveyID"][0]
        ].reset_index()

        # Create unit id
        if survey_movies_df.empty:
            # Start unit_id in 0
            new_movie_row["UnitID"] = surveys_df["SurveyID"].values[0] + "_0000"

        else:
            # Get the last unitID
            last_unitID = str(
                survey_movies_df.sort_values("UnitID").tail(1)["UnitID"].values[0]
            )[-4:]

            # Add one more to the last UnitID
            next_unitID = str(int(last_unitID) + 1).zfill(4)

            # Add one more to the last UnitID
            new_movie_row["UnitID"] = (
                surveys_df["SurveyID"].values[0] + "_" + next_unitID
            )

        # Estimate the fps and length info
        new_movie_row[["fps", "duration"]] = movie_utils.get_length(
            deployment_i, os.getcwd()
        )

        # Specify location of the concat video
        new_movie_row["concat_video"] = deployment_i

        # Specify filename of the video
        new_movie_row["filename"] = deployment_i

        # Specify the site of the deployment
        new_movie_row["SiteID"] = site_deployment

        # Specify the date of the deployment
        new_movie_row["EventDate"] = deployment_date.value

        # Save the list of goprofiles
        new_movie_row["go_pro_files"] = [movie_files_server]

        # Extract year, month and day
        new_movie_row["Year"] = new_movie_row["EventDate"][0].year
        new_movie_row["Month"] = new_movie_row["EventDate"][0].month
        new_movie_row["Day"] = new_movie_row["EventDate"][0].day

        print("The details of the new deployment are:")
        for ind in new_movie_row.T.index:
            print(ind, "-->", new_movie_row.T[0][ind])

        return new_movie_row


def upload_concat_movie(db_info_dict: dict, new_deployment_row: pd.DataFrame):
    """
    It uploads the concatenated video to the server and updates the movies csv file with the new
    information

    :param db_info_dict: a dictionary with the following keys:
    :param new_deployment_row: new deployment dataframe with the information of the new deployment
    """

    # Save to new deployment row df
    new_deployment_row["LinkToVideoFile"] = (
        "http://marine-buv.s3.ap-southeast-2.amazonaws.com/"
        + new_deployment_row["prefix_conc"][0]
    )

    # Remove temporary prefix for concatenated video and local path to concat_video
    new_movie_row = new_deployment_row.drop(["prefix_conc", "concat_video"], axis=1)

    # Load the csv with movies information
    movies_df = pd.read_csv(db_info_dict["local_movies_csv"])

    # Check the columns are the same
    diff_columns = list(
        set(movies_df.columns.sort_values().values)
        - set(new_movie_row.columns.sort_values().values)
    )

    if len(diff_columns) > 0:
        logging.error(
            f"The {diff_columns} columns are missing from the information for the new deployment."
        )
        raise

    else:
        print(
            "Uploading the concatenated movie to the server.",
            new_deployment_row["prefix_conc"][0],
        )

        # Upload movie to the s3 bucket
        server_utils.upload_file_to_s3(
            client=db_info_dict["client"],
            bucket=db_info_dict["bucket"],
            key=new_deployment_row["prefix_conc"][0],
            filename=new_deployment_row["concat_video"][0],
        )

        print("Movie uploaded to", new_deployment_row["LinkToVideoFile"])

        # Add the new row to the movies df
        movies_df = movies_df.append(new_movie_row, ignore_index=True)

        # Save the updated df locally
        movies_df.to_csv(db_info_dict["local_movies_csv"], index=False)

        # Save the updated df in the server
        server_utils.upload_file_to_s3(
            db_info_dict["client"],
            bucket=db_info_dict["bucket"],
            key=db_info_dict["server_movies_csv"],
            filename=str(db_info_dict["local_movies_csv"]),
        )

        # Remove temporary movie
        print("Movies csv file succesfully updated in the server.")


def upload_new_movies(project: p_utils.Project, db_info_dict: dict, movie_list: list):
    """
    It uploads the new movies to the SNIC server and creates new rows to be updated
    with movie metadata and saved into movies.csv

    :param db_info_dict: a dictionary with the following keys:
    :param movie_list: list of new movies that are to be added to movies.csv
    """
    # Get number of new movies to be added
    movie_folder = project.movie_folder
    number_of_movies = len(movie_list)
    # Get current movies
    movies_df = pd.read_csv(db_info_dict["local_movies_csv"])
    # Set up a new row for each new movie
    new_movie_rows_sheet = ipysheet.sheet(
        rows=number_of_movies,
        columns=movies_df.shape[1],
        column_headers=movies_df.columns.tolist(),
    )
    if len(movie_list) == 0:
        logging.error("No valid movie found to upload.")
        return
    for index, movie in enumerate(movie_list):
        if project.server == "SNIC":
            # Specify volume allocated by SNIC
            snic_path = "/mimer/NOBACKUP/groups/snic2021-6-9"
            remote_fpath = Path(f"{snic_path}/tmp_dir/", movie[1])
        else:
            remote_fpath = Path(f"{movie_folder}", movie[1])
        if os.path.exists(remote_fpath):
            logging.info(
                "Filename "
                + str(movie[1])
                + " already exists on SNIC, try again with a new file"
            )
            return
        else:
            # process video
            stem = "processed"
            p = Path(movie[0])
            processed_video_path = p.with_name(f"{p.stem}_{stem}{p.suffix}").name
            logging.info("Movie to be uploaded: " + processed_video_path)
            ffmpeg.input(p).output(
                processed_video_path,
                crf=22,
                pix_fmt="yuv420p",
                vcodec="libx264",
                threads=4,
            ).run(capture_stdout=True, capture_stderr=True, overwrite_output=True)

            if project.server == "SNIC":
                server_utils.upload_object_to_snic(
                    db_info_dict["sftp_client"],
                    str(processed_video_path),
                    str(remote_fpath),
                )
            elif project.server in ["LOCAL", "TEMPLATE"]:
                print(processed_video_path, remote_fpath)
                shutil.copy2(str(processed_video_path), str(remote_fpath))
            logging.info("movie uploaded\n")
        # Fetch movie metadata that can be calculated from movie file
        fps, duration = movie_utils.get_fps_duration(movie[0])
        movie_id = str(max(movies_df["movie_id"]) + 1)
        ipysheet.cell(index, 0, movie_id)
        ipysheet.cell(index, 1, movie[1])
        ipysheet.cell(index, 2, "-")
        ipysheet.cell(index, 3, "-")
        ipysheet.cell(index, 4, "-")
        ipysheet.cell(index, 5, fps)
        ipysheet.cell(index, 6, duration)
        ipysheet.cell(index, 7, "-")
        ipysheet.cell(index, 8, "-")
    logging.info("All movies uploaded:\n")
    logging.info(
        "Complete this sheet by filling the missing info on the movie you just uploaded"
    )
    display(new_movie_rows_sheet)
    return new_movie_rows_sheet


def add_new_rows_to_csv(db_info_dict: dict, new_movie_rows_sheet):
    """
    Save new rows into existing movies csv by concatenating old and new rows
    """

    movies_df = pd.read_csv(db_info_dict["local_movies_csv"])
    new_movie_rows_df = ipysheet.to_dataframe(new_movie_rows_sheet)
    df_with_new_rows = pd.concat([movies_df, new_movie_rows_df], ignore_index=True)
    return df_with_new_rows


def choose_new_videos_to_upload():
    """
    Simple widget for uploading videos from a file browser.
    returns the list of movies to be added.
    Supports multi-select file uploads
    """

    movie_list = []

    fc = FileChooser()
    fc.title = "First choose your directory of interest"
    " and then the movies you would like to upload"
    logging.info("Choose the file that you want to upload: ")

    def change_dir(chooser):
        sel.options = os.listdir(chooser.selected)
        fc.children[1].children[2].layout.display = "none"
        sel.layout.visibility = "visible"

    fc.register_callback(change_dir)

    sel = widgets.SelectMultiple(options=os.listdir(fc.selected))

    display(fc)
    display(sel)

    sel.layout.visibility = "hidden"

    button_add = widgets.Button(description="Add selected file")
    output_add = widgets.Output()

    logging.info(
        "Showing paths to the selected movies:\nRerun cell to reset\n--------------"
    )

    display(button_add, output_add)

    def on_button_add_clicked(b):
        with output_add:
            if sel.value is not None:
                for movie in sel.value:
                    if Path(movie).suffix in [".mp4", ".mov"]:
                        movie_list.append([Path(fc.selected, movie), movie])
                        logging.info(Path(fc.selected, movie))
                    else:
                        logging.error("Invalid file extension")
                    fc.reset()

    button_add.on_click(on_button_add_clicked)
    return movie_list
