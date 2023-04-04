# base imports
import os
import requests
import random
import pandas as pd
import numpy as np
import json
import logging
from io import BytesIO
from base64 import b64encode

# module imports
import kso_utils.db_utils as db_utils
from kso_utils.koster_utils import filter_bboxes, process_clips_koster

# from kso_utils.spyfish_utils import process_clips_spyfish
import kso_utils.project_utils as project_utils
import kso_utils.zooniverse_utils as zoo_utils

# widget imports
from IPython.display import HTML, display, clear_output
import ipywidgets as widgets
from PIL import Image as PILImage, ImageDraw
import imagesize
from jupyter_bbox_widget import BBoxWidget


# util imports
from kso_utils.db_utils import create_connection
from kso_utils.zooniverse_utils import populate_agg_annotations

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# Specify volume allocated by SNIC
snic_path = "/mimer/NOBACKUP/groups/snic2022-22-1210"


#### Set up ####
def choose_agg_parameters(subject_type: str = "clip", full_description: bool = True):
    """
    > This function creates a set of sliders that allow you to set the parameters for the aggregation
    algorithm

    :param subject_type: The type of subject you are aggregating. This can be either "frame" or "video"
    :type subject_type: str
    :return: the values of the sliders.
        Aggregation threshold: (0-1) Minimum proportion of citizen scientists that agree in their classification of the clip/frame.
        Min numbers of users: Minimum number of citizen scientists that need to classify the clip/frame.
        Object threshold (0-1): Minimum proportion of citizen scientists that agree that there is at least one object in the frame.
        IOU Epsilon (0-1): Minimum area of overlap among the classifications provided by the citizen scientists so that they will be considered to be in the same cluster.
        Inter user agreement (0-1): The minimum proportion of users inside a given cluster that must agree on the frame annotation for it to be accepted.
    """
    agg_users = widgets.FloatSlider(
        value=0.8,
        min=0,
        max=1.0,
        step=0.1,
        description="Aggregation threshold:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
    )
    # Create HTML widget for description
    description_widget = HTML(
        f"<p>Minimum proportion of citizen scientists that agree in their classification of the {subject_type}.</p>"
    )
    # Display both widgets in a VBox
    display(agg_users)
    if full_description:
        display(description_widget)
    min_users = widgets.IntSlider(
        value=3,
        min=1,
        max=15,
        step=1,
        description="Min numbers of users:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
    )
    # Create HTML widget for description
    description_widget = HTML(
        f"<p>Minimum number of citizen scientists that need to classify the {subject_type}.</p>"
    )
    # Display both widgets in a VBox
    display(min_users)
    if full_description:
        display(description_widget)
    if subject_type == "frame":
        agg_obj = widgets.FloatSlider(
            value=0.8,
            min=0,
            max=1.0,
            step=0.1,
            description="Object threshold:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".1f",
            display="flex",
            flex_flow="column",
            align_items="stretch",
            style={"description_width": "initial"},
        )
        # Create HTML widget for description
        description_widget = HTML(
            "<p>Minimum proportion of citizen scientists that agree that there is at least one object in the frame.</p>"
        )
        # Display both widgets in a VBox
        display(agg_obj)
        if full_description:
            display(description_widget)
        agg_iou = widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1.0,
            step=0.1,
            description="IOU Epsilon:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".1f",
            display="flex",
            flex_flow="column",
            align_items="stretch",
            style={"description_width": "initial"},
        )
        # Create HTML widget for description
        description_widget = HTML(
            "<p>Minimum area of overlap among the citizen science classifications to be considered as being in the same cluster.</p>"
        )
        # Display both widgets in a VBox
        display(agg_iou)
        if full_description:
            display(description_widget)
        agg_iua = widgets.FloatSlider(
            value=0.8,
            min=0,
            max=1.0,
            step=0.1,
            description="Inter user agreement:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".1f",
            display="flex",
            flex_flow="column",
            align_items="stretch",
            style={"description_width": "initial"},
        )
        # Create HTML widget for description
        description_widget = HTML(
            "<p>The minimum proportion of users inside a given cluster that must agree on the frame annotation for it to be accepted.</p>"
        )
        # Display both widgets in a VBox
        display(agg_iua)
        if full_description:
            display(description_widget)
        return agg_users, min_users, agg_obj, agg_iou, agg_iua
    else:
        return agg_users, min_users


def choose_workflows(workflows_df: pd.DataFrame):
    """
    It creates a dropdown menu for the user to choose a workflow name, a dropdown menu for the user to
    choose a subject type, and a dropdown menu for the user to choose a workflow version

    :param workflows_df: a dataframe containing the workflows you want to choose from
    :type workflows_df: pd.DataFrame
    """

    layout = widgets.Layout(width="auto", height="40px")  # set width and height

    # Display the names of the workflows
    workflow_name = widgets.Dropdown(
        options=workflows_df.display_name.unique().tolist(),
        value=workflows_df.display_name.unique().tolist()[0],
        description="Workflow name:",
        disabled=False,
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
        layout=layout,
    )

    # Display the type of subjects
    subj_type = widgets.Dropdown(
        options=["frame", "clip"],
        value="clip",
        description="Subject type:",
        disabled=False,
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
        layout=layout,
    )

    workflow_version, versions = choose_w_version(workflows_df, workflow_name.value)

    def on_change(change):
        with out:
            if change["name"] == "value":
                clear_output()
                workflow_version.options = choose_w_version(
                    workflows_df, change["new"]
                )[1]
                workflow_name.observe(on_change)

    out = widgets.Output()
    display(out)

    workflow_name.observe(on_change)
    return workflow_name, subj_type, workflow_version


class WidgetMaker(widgets.VBox):
    def __init__(self, workflows_df: pd.DataFrame):
        """
        The function creates a widget that allows the user to select which workflows to run

        :param workflows_df: the dataframe of workflows
        """
        self.workflows_df = workflows_df
        self.widget_count = widgets.BoundedIntText(
            value=0,
            min=0,
            max=100,
            description="Number of workflows:",
            display="flex",
            flex_flow="column",
            align_items="stretch",
            style={"description_width": "initial"},
        )
        self.bool_widget_holder = widgets.HBox(
            layout=widgets.Layout(
                width="70%", display="inline-flex", flex_flow="row wrap"
            )
        )
        children = [
            self.widget_count,
            self.bool_widget_holder,
        ]
        self.widget_count.observe(self._add_bool_widgets, names=["value"])
        super().__init__(children=children)

    def _add_bool_widgets(self, widg):
        num_bools = widg["new"]
        new_widgets = []
        for _ in range(num_bools):
            new_widget = choose_workflows(self.workflows_df)
            for wdgt in new_widget:
                wdgt.description = wdgt.description + f" #{_}"
            new_widgets.extend(new_widget)
        self.bool_widget_holder.children = tuple(new_widgets)

    @property
    def checks(self):
        return {w.description: w.value for w in self.bool_widget_holder.children}


def choose_w_version(workflows_df: pd.DataFrame, workflow_id: str):
    """
    It takes a workflow ID and returns a dropdown widget with the available versions of the workflow

    :param workflows_df: a dataframe containing the workflows available in the Galaxy instance
    :param workflow_id: The name of the workflow you want to run
    :return: A tuple containing the widget and the list of versions available.
    """

    # Estimate the versions of the workflow available
    versions_available = (
        workflows_df[workflows_df.display_name == workflow_id].version.unique().tolist()
    )

    if len(versions_available) > 1:
        # Display the versions of the workflow available
        w_version = widgets.Dropdown(
            options=list(map(float, versions_available)),
            value=float(versions_available[0]),
            description="Minimum workflow version:",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
            style={"description_width": "initial"},
        )

    else:
        raise ValueError("There are no versions available for this workflow.")

    # display(w_version)
    return w_version, list(map(float, versions_available))


def get_workflow_ids(workflows_df: pd.DataFrame, workflow_names: list):
    # The function that takes a list of workflow names and returns a list of workflow
    # ids.
    return [
        workflows_df[workflows_df.display_name == wf_name].workflow_id.unique()[0]
        for wf_name in workflow_names
    ]


def get_classifications(
    workflow_dict: dict,
    workflows_df: pd.DataFrame,
    subj_type: str,
    class_df: pd.DataFrame,
    db_path: str,
    project: project_utils.Project,
):
    """
    It takes in a dictionary of workflows, a dataframe of workflows, the type of subject (frame or
    clip), a dataframe of classifications, the path to the database, and the project name. It returns a
    dataframe of classifications

    :param workflow_dict: a dictionary of the workflows you want to retrieve classifications for. The
    keys are the workflow names, and the values are the workflow IDs, workflow versions, and the minimum
    number of classifications per subject
    :type workflow_dict: dict
    :param workflows_df: the dataframe of workflows from the Zooniverse project
    :type workflows_df: pd.DataFrame
    :param subj_type: "frame" or "clip"
    :param class_df: the dataframe of classifications from the database
    :param db_path: the path to the database file
    :param project: the name of the project on Zooniverse
    :return: A dataframe with the classifications for the specified project and workflow.
    """

    names, workflow_versions = [], []
    for i in range(0, len(workflow_dict), 3):
        names.append(list(workflow_dict.values())[i])
        workflow_versions.append(list(workflow_dict.values())[i + 2])

    workflow_ids = get_workflow_ids(workflows_df, names)

    # Filter classifications of interest
    classes = []
    for id, version in zip(workflow_ids, workflow_versions):
        class_df_id = class_df[
            (class_df.workflow_id == id) & (class_df.workflow_version >= version)
        ].reset_index(drop=True)
        classes.append(class_df_id)
    classes_df = pd.concat(classes)

    # Add information about the subject
    # Create connection to db
    conn = db_utils.create_connection(db_path)

    if subj_type == "frame":
        # Query id and subject type from the subjects table
        subjects_df = pd.read_sql_query(
            "SELECT id, subject_type, \
                                        https_location, filename, frame_number, movie_id FROM subjects \
                                        WHERE subject_type=='frame'",
            conn,
        )

    else:
        # Query id and subject type from the subjects table
        subjects_df = pd.read_sql_query(
            "SELECT id, subject_type, \
                                        https_location, filename, clip_start_time, movie_id FROM subjects \
                                        WHERE subject_type=='clip'",
            conn,
        )

    # Ensure id format matches classification's subject_id
    classes_df["subject_ids"] = classes_df["subject_ids"].astype("Int64")
    subjects_df["id"] = subjects_df["id"].astype("Int64")

    # Add subject information based on subject_ids
    classes_df = pd.merge(
        classes_df,
        subjects_df,
        how="left",
        left_on="subject_ids",
        right_on="id",
    )

    if classes_df[["subject_type", "https_location"]].isna().any().any():
        # Exclude classifications from missing subjects
        filtered_class_df = classes_df.dropna(
            subset=["subject_type", "https_location"], how="any"
        ).reset_index(drop=True)

        # Report on the issue
        logging.info(
            f"There are {(classes_df.shape[0]-filtered_class_df.shape[0])}"
            f" classifications out of {classes_df.shape[0]}"
            f" missing subject info. Maybe the subjects have been removed from Zooniverse?"
        )

        classes_df = filtered_class_df

    logging.info(
        f"{classes_df.shape[0]} Zooniverse classifications have been retrieved"
    )

    return classes_df


def aggregate_labels(raw_class_df: pd.DataFrame, agg_users: float, min_users: int):
    """
    > This function takes a dataframe of classifications and returns a dataframe of classifications that
    have been filtered by the number of users that classified each subject and the proportion of users
    that agreed on their annotations

    :param raw_class_df: the dataframe of all the classifications
    :param agg_users: the proportion of users that must agree on a classification for it to be included
    in the final dataset
    :param min_users: The minimum number of users that must have classified a subject for it to be
    included in the final dataset
    :return: a dataframe with the aggregated labels.
    """
    # Calculate the number of users that classified each subject
    raw_class_df["n_users"] = raw_class_df.groupby("subject_ids")[
        "classification_id"
    ].transform("nunique")

    # Select classifications with at least n different user classifications
    raw_class_df = raw_class_df[raw_class_df.n_users >= min_users].reset_index(
        drop=True
    )

    # Calculate the proportion of unique classifications (it can have multiple annotations) per subject
    raw_class_df["class_n"] = raw_class_df.groupby(["subject_ids", "label"])[
        "classification_id"
    ].transform("nunique")

    # Calculate the proportion of users that agreed on their annotations
    raw_class_df["class_prop"] = raw_class_df.class_n / raw_class_df.n_users

    # Select annotations based on agreement threshold
    agg_class_df = raw_class_df[raw_class_df.class_prop >= agg_users].reset_index(
        drop=True
    )

    # Calculate the proportion of unique classifications aggregated per subject
    agg_class_df["class_n_agg"] = agg_class_df.groupby(["subject_ids"])[
        "label"
    ].transform("nunique")

    return agg_class_df


def aggregate_classifications(
    df: pd.DataFrame, subj_type: str, project: project_utils.Project, agg_params: list
):
    """
    We take the raw classifications and process them to get the aggregated labels

    :param df: the raw classifications dataframe
    :param subj_type: the type of subject, either "frame" or "clip"
    :param project: the project object
    :param agg_params: list of parameters for the aggregation
    :return: the aggregated classifications and the raw classifications.
    """

    logging.info("Aggregating the classifications")

    # We take the raw classifications and process them to get the aggregated labels.
    if subj_type == "frame":
        # Get the aggregation parameters
        if not isinstance(agg_params, list):
            agg_users, min_users, agg_obj, agg_iou, agg_iua = [
                i.value for i in agg_params
            ]
        else:
            agg_users, min_users, agg_obj, agg_iou, agg_iua = agg_params

        # Report selected parameters
        logging.info(
            f"Aggregation parameters are: Agg. threshold "
            f"{agg_users} "
            f"Min. users "
            f"{min_users} "
            f"Obj threshold "
            f"{agg_obj} "
            f"IOU "
            f"{agg_iou} "
            f"Int. agg. "
            f"{agg_iua} "
        )
        # Process the raw classifications
        raw_class_df = process_frames(df, project.Project_name)

        # Aggregate frames based on their labels
        agg_labels_df = aggregate_labels(raw_class_df, agg_users, min_users)

        # Get rid of the "empty" labels if other species are among the volunteer consensus
        agg_labels_df = agg_labels_df[
            ~((agg_labels_df["class_n_agg"] > 1) & (agg_labels_df["label"] == "empty"))
        ]

        # Select frames aggregated only as empty
        agg_labels_df_empty = agg_labels_df[agg_labels_df["label"] == "empty"]
        agg_labels_df_empty = agg_labels_df_empty.rename(
            columns={"frame_number": "start_frame"}
        )
        agg_labels_df_empty = agg_labels_df_empty[
            ["label", "subject_ids", "x", "y", "w", "h"]
        ]

        # Temporary exclude frames aggregated as empty
        agg_labels_df = agg_labels_df[agg_labels_df["label"] != "empty"]

        # Map the position of the annotation parameters
        col_list = list(agg_labels_df.columns)
        x_pos, y_pos, w_pos, h_pos, user_pos, subject_id_pos = (
            col_list.index("x"),
            col_list.index("y"),
            col_list.index("w"),
            col_list.index("h"),
            col_list.index("user_name"),
            col_list.index("subject_ids"),
        )

        # Get prepared annotations
        new_rows = []

        if agg_labels_df["frame_number"].isnull().all():
            group_cols = ["subject_ids", "label"]
        else:
            group_cols = ["subject_ids", "label", "frame_number"]

        for name, group in agg_labels_df.groupby(group_cols):
            if "frame_number" in group_cols:
                subj_id, label, start_frame = name
                total_users = agg_labels_df[
                    (agg_labels_df.subject_ids == subj_id)
                    & (agg_labels_df.label == label)
                    & (agg_labels_df.frame_number == start_frame)
                ]["user_name"].nunique()
            else:
                subj_id, label = name
                start_frame = np.nan
                total_users = agg_labels_df[
                    (agg_labels_df.subject_ids == subj_id)
                    & (agg_labels_df.label == label)
                ]["user_name"].nunique()

            # Filter bboxes using IOU metric (essentially a consensus metric)
            # Keep only bboxes where mean overlap exceeds this threshold
            indices, new_group = filter_bboxes(
                total_users=total_users,
                users=[i[user_pos] for i in group.values],
                bboxes=[
                    np.array([i[x_pos], i[y_pos], i[w_pos], i[h_pos]])
                    for i in group.values
                ],
                obj=agg_obj,
                eps=agg_iou,
                iua=agg_iua,
            )

            subject_ids = [i[subject_id_pos] for i in group.values[indices]]

            for ix, box in zip(subject_ids, new_group):
                new_rows.append(
                    (
                        label,
                        start_frame,
                        ix,
                    )
                    + tuple(box)
                )

        agg_class_df = pd.DataFrame(
            new_rows,
            columns=[
                "label",
                "start_frame",
                "subject_ids",
                "x",
                "y",
                "w",
                "h",
            ],
        )

        agg_class_df["subject_type"] = "frame"
        agg_class_df["label"] = agg_class_df["label"].apply(
            lambda x: x.split("(")[0].strip()
        )

        # Add the frames aggregated as "empty"
        agg_class_df = pd.concat([agg_class_df, agg_labels_df_empty])

        # Select the aggregated labels
        agg_class_df = agg_class_df[
            ["subject_ids", "label", "x", "y", "w", "h"]
        ].drop_duplicates()

        # Add the http info
        agg_class_df = pd.merge(
            agg_class_df,
            raw_class_df[
                [
                    "subject_ids",
                    "https_location",
                    "subject_type",
                    "workflow_id",
                    "workflow_name",
                    "workflow_version",
                ]
            ].drop_duplicates(),
            how="left",
            on="subject_ids",
        )

    else:
        # Get the aggregation parameters
        if not isinstance(agg_params, list):
            agg_users, min_users = [i.value for i in agg_params]
        else:
            agg_users, min_users = agg_params

        # Process the raw classifications
        raw_class_df = process_clips(df, project)

        # aggregate clips based on their labels
        agg_class_df = aggregate_labels(raw_class_df, agg_users, min_users)

        # Extract the median of the second where the animal/object is and number of animals
        agg_class_df = agg_class_df.groupby(
            [
                "subject_ids",
                "https_location",
                "subject_type",
                "label",
                "workflow_id",
                "workflow_name",
                "workflow_version",
            ],
            as_index=False,
        )
        agg_class_df = pd.DataFrame(
            agg_class_df[["how_many", "first_seen"]].median().round(0)
        )

    # Add username info to raw class
    raw_class_df = pd.merge(
        raw_class_df,
        df[["classification_id", "user_name"]],
        how="left",
        on="classification_id",
    )

    logging.info(
        f"{agg_class_df.shape[0]}"
        "classifications aggregated out of"
        f"{df.subject_ids.nunique()}"
        "unique subjects available"
    )

    return agg_class_df, raw_class_df


def process_clips(df: pd.DataFrame, project: project_utils.Project):
    """
    This function takes a dataframe of classifications and returns a dataframe of annotations

    :param df: the dataframe of classifications
    :type df: pd.DataFrame
    :param project: the name of the project you want to download data from
    :return: A dataframe with the classification_id, label, how_many, first_seen, https_location,
    subject_type, and subject_ids.
    """

    # Create an empty list
    rows_list = []

    # Loop through each classification submitted by the users
    for index, row in df.iterrows():
        # Load annotations as json format
        annotations = json.loads(row["annotations"])

        # Select the information from the species identification task
        if project.Zooniverse_number == 9747:
            rows_list = process_clips_koster(
                annotations, row["classification_id"], rows_list
            )

        # Check if the Zooniverse project is the Spyfish
        if project.Project_name == "Spyfish_Aotearoa":
            rows_list = process_clips_spyfish(
                annotations, row["classification_id"], rows_list
            )

        # Process clips as the default method
        else:
            rows_list = zoo_utils.process_clips_template(
                annotations, row["classification_id"], rows_list
            )

    # Create a data frame with annotations as rows
    annot_df = pd.DataFrame(
        rows_list, columns=["classification_id", "label", "first_seen", "how_many"]
    )

    # Specify the type of columns of the df
    annot_df["how_many"] = pd.to_numeric(annot_df["how_many"])
    annot_df["first_seen"] = pd.to_numeric(annot_df["first_seen"])

    # Add subject id to each annotation
    annot_df = pd.merge(
        annot_df,
        df.drop(columns=["annotations"]),
        how="left",
        on="classification_id",
    )

    # Select only relevant columns
    annot_df = annot_df[
        [
            "classification_id",
            "label",
            "how_many",
            "first_seen",
            "https_location",
            "subject_type",
            "subject_ids",
            "workflow_id",
            "workflow_name",
            "workflow_version",
        ]
    ]

    return pd.DataFrame(annot_df)


def launch_table(agg_class_df: pd.DataFrame, subject_type: str):
    """
    It takes in a dataframe of aggregated classifications and a subject type, and returns a dataframe
    with the columns "subject_ids", "label", "how_many", and "first_seen"

    :param agg_class_df: the dataframe that you want to launch
    :param subject_type: "clip" or "subject"
    """
    if subject_type == "clip":
        a = agg_class_df[["subject_ids", "label", "how_many", "first_seen"]]
    else:
        a = agg_class_df

    return a


def process_frames(df: pd.DataFrame, project_name: str):
    """
    It takes a dataframe of classifications and returns a dataframe of annotations

    :param df: the dataframe containing the classifications
    :type df: pd.DataFrame
    :param project_name: The name of the project you want to download data from
    :return: A dataframe with the following columns:
        classification_id, x, y, w, h, label, https_location, filename, subject_type, subject_ids,
    frame_number, user_name, movie_id
    """

    # Create an empty list
    rows_list = []

    # Loop through each classification submitted by the users and flatten them
    for index, row in df.iterrows():
        # Load annotations as json format
        annotations = json.loads(row["annotations"])

        # Select the information from all the labelled animals (e.g. task = T0)
        for ann_i in annotations:
            if ann_i["task"] == "T0":
                if ann_i["value"] == []:
                    # Specify the frame was classified as empty
                    choice_i = {
                        "classification_id": row["classification_id"],
                        "x": None,
                        "y": None,
                        "w": None,
                        "h": None,
                        "label": "empty",
                    }
                    rows_list.append(choice_i)

                else:
                    # Select each species annotated and flatten the relevant answers
                    for i in ann_i["value"]:
                        choice_i = {
                            "classification_id": row["classification_id"],
                            "x": int(i["x"]) if "x" in i else None,
                            "y": int(i["y"]) if "y" in i else None,
                            "w": int(i["width"]) if "width" in i else None,
                            "h": int(i["height"]) if "height" in i else None,
                            "label": str(i["tool_label"])
                            if "tool_label" in i
                            else None,
                        }
                        rows_list.append(choice_i)

    # Create a data frame with annotations as rows
    flat_annot_df = pd.DataFrame(
        rows_list, columns=["classification_id", "x", "y", "w", "h", "label"]
    )

    # Add other classification information to the flatten classifications
    annot_df = pd.merge(
        flat_annot_df,
        df,
        how="left",
        on="classification_id",
    )

    # Select only relevant columns
    annot_df = annot_df[
        [
            "classification_id",
            "x",
            "y",
            "w",
            "h",
            "label",
            "https_location",
            "filename",
            "subject_type",
            "subject_ids",
            "frame_number",
            "user_name",
            "movie_id",
            "workflow_version",
            "workflow_name",
            "workflow_id",
        ]
    ]

    return pd.DataFrame(annot_df)


def draw_annotations_in_frame(im: PILImage.Image, class_df_subject: pd.DataFrame):
    """
    > The function takes an image and a dataframe of annotations and returns the image with the
    annotations drawn on it

    :param im: the image object of type PILImage
    :param class_df_subject: a dataframe containing the annotations for a single subject
    :return: The image with the annotations
    """
    # Calculate image size
    dw, dh = im._size

    # Draw rectangles of each annotation
    img1 = ImageDraw.Draw(im)

    # Merge annotation info into a tuple
    class_df_subject["vals"] = class_df_subject[["x", "y", "w", "h"]].values.tolist()

    for index, row in class_df_subject.iterrows():
        # Specify the vals object
        vals = row.vals

        # Adjust annotantions to image size
        vals_adjusted = tuple(
            [
                int(vals[0]),
                int(vals[1]),
                int((vals[0] + vals[2])),
                int((vals[1] + vals[3])),
            ]
        )

        # Draw annotation
        img1.rectangle(vals_adjusted, width=2)

    return im


def view_subject(subject_id: int, class_df: pd.DataFrame, subject_type: str):
    """
    It takes a subject id, a dataframe containing the annotations for that subject, and the type of
    subject (clip or frame) and returns an HTML object that can be displayed in a notebook

    :param subject_id: The subject ID of the subject you want to view
    :type subject_id: int
    :param class_df: The dataframe containing the annotations for the class of interest
    :type class_df: pd.DataFrame
    :param subject_type: The type of subject you want to view. This can be either "clip" or "frame"
    :type subject_type: str
    """
    if subject_id in class_df.subject_ids.tolist():
        # Select the subject of interest
        class_df_subject = class_df[class_df.subject_ids == subject_id].reset_index(
            drop=True
        )

        # Get the location of the subject
        subject_location = class_df_subject["https_location"].unique()[0]

    else:
        raise Exception("The reference data does not contain media for this subject.")

    if len(subject_location) == 0:
        raise Exception("Subject not found in provided annotations")

    # Get the HTML code to show the selected subject
    if subject_type == "clip":
        html_code = f"""
        <html>
        <div style="display: flex; justify-content: space-around">
        <div>
          <video width=500 controls>
          <source src={subject_location} type="video/mp4">
        </video>
        </div>
        <div>{class_df_subject[['label','first_seen','how_many']].value_counts().sort_values(ascending=False).to_frame().to_html()}</div>
        </div>
        </html>"""

    elif subject_type == "frame":
        # Read image
        response = requests.get(subject_location)
        im = PILImage.open(BytesIO(response.content))

        # if label is not empty draw rectangles
        if class_df_subject.label.unique()[0] != "empty":
            # Create a temporary image with the annotations drawn on it
            im = draw_annotations_in_frame(im, class_df_subject)

        # Remove previous temp image if exist
        if os.access(".", os.W_OK):
            temp_image_path = "temp.jpg"
        else:
            temp_image_path = f"{snic_path}/tmp_dir/temp.jpg"

        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        # Save the new image
        im.save(temp_image_path)

        # Load image data (used to enable viewing in Colab)
        img = open(temp_image_path, "rb").read()
        data_url = "data:image/jpeg;base64," + b64encode(img).decode()

        html_code = f"""
        <html>
        <div style="display: flex; justify-content: space-around">
        <div>
          <img src={data_url} type="image/jpeg" width=500>
        </img>
        </div>
        <div>{class_df_subject[['label','colour']].value_counts().sort_values(ascending=False).to_frame().to_html()}</div>
        </div>
        </html>"""
    else:
        Exception("Subject type not supported.")
    return HTML(html_code)


def launch_viewer(class_df: pd.DataFrame, subject_type: str):
    """
    > This function takes a dataframe of classifications and a subject type (frame or video) and
    displays a dropdown menu of subjects of that type. When a subject is selected, it displays the
    subject and the classifications for that subject

    :param class_df: The dataframe containing the classifications
    :type class_df: pd.DataFrame
    :param subject_type: The type of subject you want to view. This can be either "frame" or "video"
    :type subject_type: str
    """

    # If subject is frame assign a color to each label
    if subject_type == "frame":
        # Create a list of unique labels
        list_labels = class_df.label.unique().tolist()

        # Generate a list of random colors for each label
        random_color_list = []
        for index, item in enumerate(list_labels):
            random_color_list = random_color_list + [
                "#" + "".join([random.choice("ABCDEF0123456789") for i in range(6)])
            ]

        # Add a column with the color for each label
        class_df["colour"] = class_df.apply(
            lambda row: random_color_list[list_labels.index(row.label)], axis=1
        )

    # Select the subject
    options = tuple(
        class_df[class_df["subject_type"] == subject_type]["subject_ids"]
        .apply(int)
        .apply(str)
        .unique()
    )
    subject_widget = widgets.Dropdown(
        options=options,
        description="Subject id:",
        ensure_option=True,
        disabled=False,
    )

    main_out = widgets.Output()
    display(subject_widget, main_out)

    # Display the subject and classifications on change
    def on_change(change):
        with main_out:
            a = view_subject(int(change["new"]), class_df, subject_type)
            clear_output()
            display(a)

    subject_widget.observe(on_change, names="value")


def explore_classifications_per_subject(class_df: pd.DataFrame, subject_type: str):
    """
    > This function takes a dataframe of classifications and a subject type (clip or frame) and displays
    the classifications for a given subject

    :param class_df: the dataframe of classifications
    :type class_df: pd.DataFrame
    :param subject_type: "clip" or "frame"
    """

    # Select the subject
    subject_widget = widgets.Dropdown(
        options=tuple(class_df.subject_ids.apply(int).apply(str).unique()),
        description="Subject id:",
        ensure_option=True,
        disabled=False,
    )

    main_out = widgets.Output()
    display(subject_widget, main_out)

    # Display the subject and classifications on change
    def on_change(change):
        with main_out:
            a = class_df[class_df.subject_ids == int(change["new"])]
            if subject_type == "clip":
                a = a[
                    [
                        "classification_id",
                        "user_name",
                        "label",
                        "how_many",
                        "first_seen",
                    ]
                ]
            else:
                a = a[
                    [
                        "x",
                        "y",
                        "w",
                        "h",
                        "label",
                        "https_location",
                        "subject_ids",
                        "frame_number",
                        "movie_id",
                    ]
                ]
            clear_output()
            display(a)

    subject_widget.observe(on_change, names="value")


def encode_image(filepath):
    """
    It takes a filepath to an image, opens the image, reads the bytes, encodes the bytes as base64, and
    returns the encoded string

    :param filepath: The path to the image file
    :return: the base64 encoding of the image.
    """
    with open(filepath, "rb") as f:
        image_bytes = f.read()
    encoded = str(b64encode(image_bytes), "utf-8")
    return "data:image/jpg;base64," + encoded


def get_annotations_viewer(data_path: str, species_list: list):
    """
    It takes a path to a folder containing images and annotations, and a list of species names, and
    returns a widget that allows you to view the images and their annotations, and to edit the
    annotations

    :param data_path: the path to the data folder
    :type data_path: str
    :param species_list: a list of species names
    :type species_list: list
    :return: A VBox widget containing a progress bar and a BBoxWidget.
    """
    image_path = os.path.join(data_path, "images")
    annot_path = os.path.join(data_path, "labels")

    images = sorted(
        [
            f
            for f in os.listdir(image_path)
            if os.path.isfile(os.path.join(image_path, f))
        ]
    )
    annotations = sorted(
        [
            f
            for f in os.listdir(annot_path)
            if os.path.isfile(os.path.join(annot_path, f))
        ]
    )

    if any([len(images), len(annotations)]) == 0:
        logging.error("No annotations to display")
        return None

    # a progress bar to show how far we got
    w_progress = widgets.IntProgress(value=0, max=len(images), description="Progress")
    # the bbox widget
    image = os.path.join(image_path, images[0])
    width, height = imagesize.get(image)
    label_file = annotations[w_progress.value]
    bboxes = []
    labels = []
    with open(os.path.join(annot_path, label_file), "r") as f:
        for line in f:
            s = line.split(" ")
            labels.append(s[0])

            left = (float(s[1]) - (float(s[3]) / 2)) * width
            top = (float(s[2]) - (float(s[4]) / 2)) * height

            bboxes.append(
                {
                    "x": left,
                    "y": top,
                    "width": float(s[3]) * width,
                    "height": float(s[4]) * height,
                    "label": species_list[int(s[0])],
                }
            )
    w_bbox = BBoxWidget(image=encode_image(image), classes=species_list)

    # here we assign an empty list to bboxes but
    # we could also run a detection model on the file
    # and use its output for creating inital bboxes
    w_bbox.bboxes = bboxes

    # combine widgets into a container
    w_container = widgets.VBox(
        [
            w_progress,
            w_bbox,
        ]
    )

    def on_button_clicked(b):
        w_progress.value = 0
        image = os.path.join(image_path, images[0])
        width, height = imagesize.get(image)
        label_file = annotations[w_progress.value]
        bboxes = []
        labels = []
        with open(os.path.join(annot_path, label_file), "r") as f:
            for line in f:
                s = line.split(" ")
                labels.append(s[0])

                left = (float(s[1]) - (float(s[3]) / 2)) * width
                top = (float(s[2]) - (float(s[4]) / 2)) * height

                bboxes.append(
                    {
                        "x": left,
                        "y": top,
                        "width": float(s[3]) * width,
                        "height": float(s[4]) * height,
                        "label": species_list[int(s[0])],
                    }
                )
        w_bbox.image = encode_image(image)

        # here we assign an empty list to bboxes but
        # we could also run a detection model on the file
        # and use its output for creating inital bboxes
        w_bbox.bboxes = bboxes
        w_container.children = tuple(list(w_container.children[1:]))
        b.close()

    # when Skip button is pressed we move on to the next file
    def on_skip():
        w_progress.value += 1
        if w_progress.value == len(annotations):
            button = widgets.Button(
                description="Click to restart.",
                disabled=False,
                display="flex",
                flex_flow="column",
                align_items="stretch",
            )
            if isinstance(w_container.children[0], widgets.Button):
                w_container.children = tuple(list(w_container.children[1:]))
            w_container.children = tuple([button] + list(w_container.children))
            button.on_click(on_button_clicked)

        # open new image in the widget
        else:
            image_file = images[w_progress.value]
            image_p = os.path.join(image_path, image_file)
            width, height = imagesize.get(image_p)
            w_bbox.image = encode_image(image_p)
            label_file = annotations[w_progress.value]
            bboxes = []
            with open(os.path.join(annot_path, label_file), "r") as f:
                for line in f:
                    s = line.split(" ")
                    left = (float(s[1]) - (float(s[3]) / 2)) * width
                    top = (float(s[2]) - (float(s[4]) / 2)) * height
                    bboxes.append(
                        {
                            "x": left,
                            "y": top,
                            "width": float(s[3]) * width,
                            "height": float(s[4]) * height,
                            "label": species_list[int(s[0])],
                        }
                    )

            # here we assign an empty list to bboxes but
            # we could also run a detection model on the file
            # and use its output for creating initial bboxes
            w_bbox.bboxes = bboxes

    w_bbox.on_skip(on_skip)

    # when Submit button is pressed we save current annotations
    # and then move on to the next file
    def on_submit():
        image_file = images[w_progress.value]
        width, height = imagesize.get(os.path.join(image_path, image_file))
        # save annotations for current image
        open(os.path.join(annot_path, label_file), "w").write(
            "\n".join(
                [
                    "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                        species_list.index(
                            i["label"]
                        ),  # single class vs multiple classes
                        min((i["x"] + i["width"] / 2) / width, 1.0),
                        min((i["y"] + i["height"] / 2) / height, 1.0),
                        min(i["width"] / width, 1.0),
                        min(i["height"] / height, 1.0),
                    )
                    for i in w_bbox.bboxes
                ]
            )
        )
        # move on to the next file
        on_skip()

    w_bbox.on_submit(on_submit)

    return w_container


def format_to_gbif_occurence(
    df: pd.DataFrame,
    classified_by: str,
    subject_type: str,
    db_info_dict: dict,
    project: project_utils.Project,
    zoo_info_dict: dict,
):
    """
    > This function takes a df of biological observations classified by citizen scientists, biologists or ML algorithms and returns a df of species occurrences to publish in GBIF/OBIS.
    :param df: the dataframe containing the aggregated classifications
    :param classified_by: the entity who classified the object of interest, either "citizen_scientists", "biologists" or "ml_algorithms"
    :param subject_type: str,
    :param db_info_dict: a dictionary containing the path to the database and the database name
    :param project: the project object
    :param zoo_info_dict: dictionary with the workflow/subjects/classifications retrieved from Zooniverse project
    :return: a df of species occurrences to publish in GBIF/OBIS.
    """

    # If classifications have been created by citizen scientists
    if classified_by == "citizen_scientists":
        #### Retrieve subject information #####
        # Create connection to db
        conn = create_connection(db_info_dict["db_path"])

        # Add annotations to db
        populate_agg_annotations(df, subject_type, project)

        # Retrieve list of subjects
        subjects_df = pd.read_sql_query(
            "SELECT id, clip_start_time, frame_number, movie_id FROM subjects",
            conn,
        )

        # Ensure subject_ids format is int
        df["subject_ids"] = df["subject_ids"].astype(int)
        subjects_df["id"] = subjects_df["id"].astype(int)

        # Combine the aggregated clips and subjects dataframes
        comb_df = pd.merge(
            df, subjects_df, how="left", left_on="subject_ids", right_on="id"
        ).drop(columns=["id"])

        #### Retrieve movie and site information #####
        # Query info about the movie of interest
        movies_df = pd.read_sql_query("SELECT * FROM movies", conn)

        # Add survey information as part of the movie info if spyfish
        if "local_surveys_csv" in db_info_dict.keys():
            # Read info about the movies
            movies_csv = pd.read_csv(db_info_dict["local_movies_csv"])

            # Select only movie ids and survey ids
            movies_csv = movies_csv[["movie_id", "SurveyID"]]

            # Combine the movie_id and survey information
            movies_df = pd.merge(
                movies_df, movies_csv, how="left", left_on="id", right_on="movie_id"
            ).drop(columns=["movie_id"])

            # Read info about the surveys
            surveys_df = pd.read_csv(
                db_info_dict["local_surveys_csv"], parse_dates=["SurveyStartDate"]
            )

            # Combine the movie_id and survey information
            movies_df = pd.merge(
                movies_df,
                surveys_df,
                how="left",
                left_on="SurveyID",
                right_on="SurveyID",
            )

        # Combine the aggregated clips and subjects dataframes
        comb_df = pd.merge(
            comb_df, movies_df, how="left", left_on="movie_id", right_on="id"
        ).drop(columns=["id"])

        # Query info about the sites of interest
        sites_df = pd.read_sql_query("SELECT * FROM sites", conn)

        # Combine the aggregated classifications and site information
        comb_df = pd.merge(
            comb_df, sites_df, how="left", left_on="site_id", right_on="id"
        ).drop(columns=["id"])

        #### Retrieve species/labels information #####
        # Create a df with unique workflow ids and versions of interest
        work_df = (
            df[["workflow_id", "workflow_version"]].drop_duplicates().astype("int")
        )

        # Correct for some weird zooniverse version behaviour
        work_df["workflow_version"] = work_df["workflow_version"] - 1

        # Store df of all the common names and the labels into a list of df
        commonName_labels_list = [
            get_workflow_labels(zoo_info_dict["workflows"], x, y)
            for x, y in zip(work_df["workflow_id"], work_df["workflow_version"])
        ]

        # Concatenate the dfs and select only unique common names and the labels
        commonName_labels_df = pd.concat(commonName_labels_list).drop_duplicates()

        # Rename the columns as they are the other way aorund (potentially only in Spyfish?)
        vernacularName_labels_df = commonName_labels_df.rename(
            columns={
                "commonName": "label",
                "label": "vernacularName",
            }
        )

        # Combine the labels with the commonNames of the classifications
        comb_df = pd.merge(comb_df, vernacularName_labels_df, how="left", on="label")

        # Query info about the species of interest
        species_df = pd.read_sql_query("SELECT * FROM species", conn)

        # Rename the column to match Darwin core std
        species_df = species_df.rename(
            columns={
                "label": "vernacularName",
            }
        )
        # Combine the aggregated classifications and species information
        comb_df = pd.merge(comb_df, species_df, how="left", on="vernacularName")

        #### Tidy up classifications information #####
        if subject_type == "clip":
            # Identify the second of the original movie when the species first appears
            comb_df["second_in_movie"] = (
                comb_df["clip_start_time"] + comb_df["first_seen"]
            )

        if subject_type == "frame":
            # Identify the second of the original movie when the species appears
            comb_df["second_in_movie"] = comb_df["frame_number"] * comb_df["fps"]

        # Drop the clips classified as nothing here
        comb_df = comb_df[comb_df["label"] != "NOTHINGHERE"]
        comb_df = comb_df[comb_df["label"] != "OTHER"]

        # Select the max count of each species on each movie
        comb_df = comb_df.sort_values("how_many").drop_duplicates(
            ["movie_id", "vernacularName"], keep="last"
        )

        # Rename columns to match Darwin Data Core Standards
        comb_df = comb_df.rename(
            columns={
                "created_on": "eventDate",
                "how_many": "individualCount",
            }
        )

        # Create relevant columns for GBIF
        comb_df["occurrenceID"] = (
            project.Project_name
            + "_"
            + comb_df["siteName"]
            + "_"
            + comb_df["eventDate"].astype(str)
            + "_"
            + comb_df["second_in_movie"].astype(str)
            + "_"
            + comb_df["vernacularName"].astype(str)
        )

        comb_df["basisOfRecord"] = "MachineObservation"

        # If coord uncertainity doesn't exist set to 30 metres
        comb_df["coordinateUncertaintyInMeters"] = comb_df.get(
            "coordinateUncertaintyInMeters", 30
        )

        # Select columns relevant for GBIF occurrences
        comb_df = comb_df[
            [
                "occurrenceID",
                "basisOfRecord",
                "vernacularName",
                "scientificName",
                "eventDate",
                "countryCode",
                "taxonRank",
                "kingdom",
                "decimalLatitude",
                "decimalLongitude",
                "geodeticDatum",
                "coordinateUncertaintyInMeters",
                "individualCount",
            ]
        ]

        return comb_df

    # If classifications have been created by biologists
    if classified_by == "biologists":
        logging.info("This sections is currently under development")

    # If classifications have been created by ml algorithms
    if classified_by == "ml_algorithms":
        logging.info("This sections is currently under development")
    else:
        raise ValueError(
            "Specify who classified the species of interest (citizen_scientists, biologists or ml_algorithms)"
        )


def get_workflow_labels(
    workflow_df: pd.DataFrame, workflow_id: int, workflow_version: int
):
    """
    > This function takes a df of workflows of interest and retrieves the labels and common names of the choices cit scientists have in a survey task in Zooniverse.
    the function is a modified version of the 'get_workflow_info' function by @lcjohnso
    https://github.com/zooniverse/Data-digging/blob/6e9dc5db6f6125316616c4b04ae5fc4223826a25/scripts_GeneralPython/get_workflow_info.pybiological observations classified by citizen scientists, biologists or ML algorithms and returns a df of species occurrences to publish in GBIF/OBIS.
    :param workflow_df: df of the workflows of the Zooniverse project of interest,
    :param workflow_id: integer of the workflow id of interest,
    :param workflow_version: integer of the workflow version of interest.
    :return: a df with the common name and label of the annotations for the workflow.
    """
    # initialize the output
    workflow_info = {}

    # parse the tasks column as a json so we can work with it (it just loads as a string)
    workflow_df["tasks_json"] = [json.loads(q) for q in workflow_df["tasks"]]
    workflow_df["strings_json"] = [json.loads(q) for q in workflow_df["strings"]]

    # identify the row of the workflow dataframe we want to extract
    is_theworkflow = (workflow_df["workflow_id"] == workflow_id) & (
        workflow_df["version"] == workflow_version
    )

    # extract it
    theworkflow = workflow_df[is_theworkflow]

    # pandas is a little weird about accessing stuff sometimes
    # we should only have 1 row in theworkflow but the row index will be retained
    # from the full workflow_df, so we need to figure out what it is
    i_wf = theworkflow.index[0]

    # extract the tasks as a json
    tasks = theworkflow["tasks_json"][i_wf]
    strings = theworkflow["strings_json"][i_wf]

    workflow_info = tasks.copy()

    tasknames = workflow_info.keys()
    workflow_info["tasknames"] = tasknames

    # now that we've extracted the actual task names, add the first task
    workflow_info["first_task"] = theworkflow["first_task"].values[0]

    # now join workflow structure to workflow label content for each task

    for task in tasknames:
        # Create an empty dictionary to host the dfs of interest
        label_common_name_dict = {"commonName": [], "label": []}

        # Create an empty dictionary to host the dfs of interest
        label_common_name_dict = {"commonName": [], "label": []}
        for i_c, choice in enumerate(workflow_info[task]["choices"].keys()):
            c_label = strings[workflow_info[task]["choices"][choice]["label"]]
            label_common_name_dict["commonName"].append(choice)
            label_common_name_dict["label"].append(c_label)

        if task == "T0":
            break

    return pd.DataFrame.from_dict(label_common_name_dict)
