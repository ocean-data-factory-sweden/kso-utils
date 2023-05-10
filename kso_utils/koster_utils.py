# -*- coding: utf-8 -*-
# base imports
import json
import logging
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path
from collections import Counter

# util imports
from kso_utils.project_utils import Project

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


# Function to extract metadata from subjects
def extract_metadata(subj_df: pd.DataFrame):
    """
    The function extracts metadata from a pandas DataFrame and returns two separate DataFrames, one with
    the metadata flattened and one without the metadata.
    
    :param subj_df: A pandas DataFrame containing subject data, including metadata information in JSON
    format
    :type subj_df: pd.DataFrame
    :return: The function `extract_metadata` returns two dataframes: `subj_df` and `meta_df`. `subj_df`
    is the original input dataframe with the "metadata" and "index" columns dropped, and with the index
    reset. `meta_df` is a new dataframe that contains the flattened metadata information extracted from
    the "metadata" column of the input dataframe.
    """
    # Reset index of df
    subj_df = subj_df.reset_index(drop=True).reset_index()

    # Flatten the metadata information
    meta_df = pd.json_normalize(subj_df.metadata.apply(json.loads))

    # Drop metadata and index columns from original df
    subj_df = subj_df.drop(
        columns=[
            "metadata",
            "index",
        ]
    )

    return subj_df, meta_df


# Function to process subjects uploaded automatically
def auto_subjects(subjects_df: pd.DataFrame, auto_date: str):
    """
    The function `auto_subjects` selects and extracts metadata from subjects that were automatically
    uploaded after a specified date.
    
    :param subjects_df: `subjects_df` is a pandas DataFrame containing information about subjects, such
    as their IDs, project IDs, and creation dates
    :type subjects_df: pd.DataFrame
    :param auto_date: auto_date is a string parameter that represents the date from which the function
    should select automatically uploaded frames. The function will only select frames that were created
    after this date
    :type auto_date: str
    :return: The function `auto_subjects` returns a pandas DataFrame containing metadata information
    extracted from subjects that were automatically uploaded after a specified date.
    """
    # Select automatically uploaded frames
    auto_subjects_df = subjects_df[(subjects_df["created_at"] > auto_date)]

    # Extract metadata from automatically uploaded subjects
    auto_subjects_df, auto_subjects_meta = extract_metadata(auto_subjects_df)

    # Combine metadata info with the subjects df
    auto_subjects_df = pd.concat([auto_subjects_df, auto_subjects_meta], axis=1)

    return auto_subjects_df


# Function to process subjects uploaded manually
def manual_subjects(subjects_df: pd.DataFrame, manual_date: str, auto_date: str):
    """
    The function extracts metadata from manually uploaded clips and processes it to combine with the
    subjects dataframe.
    
    :param subjects_df: A pandas DataFrame containing information about subjects, including metadata and
    creation dates
    :type subjects_df: pd.DataFrame
    :param manual_date: The date from which to start selecting clips uploaded manually
    :type manual_date: str
    :param auto_date: It seems like the parameter auto_date is missing from the code snippet. Can you
    provide more information on what this parameter represents?
    :type auto_date: str
    :return: a pandas DataFrame containing information about clips that were uploaded manually, along
    with their metadata and processed information.
    """
    # Select clips uploaded manually
    man_clips_df = (
        subjects_df[
            (subjects_df["metadata"].str.contains(".mp4"))
            & (subjects_df["created_at"].between(manual_date, auto_date))
        ]
        .reset_index(drop=True)
        .reset_index()
    )

    # Specify the type of subject
    man_clips_df["subject_type"] = "clip"

    # Extract metadata from manually uploaded clips
    man_clips_df, man_clips_meta = extract_metadata(man_clips_df)

    # Process the metadata of manually uploaded clips
    man_clips_meta = process_manual_clips(man_clips_meta)

    # Combine metadata info with the subjects df
    man_clips_df = pd.concat([man_clips_df, man_clips_meta], axis=1)

    return man_clips_df


# Function to process the metadata of clips that were uploaded manually
def process_manual_clips(meta_df: pd.DataFrame):
    """
    The function processes metadata of manual clips by extracting relevant information such as clip
    start and end times and the filename of the original movie.
    
    :param meta_df: The input parameter `meta_df` is a Pandas DataFrame containing metadata information
    about video clips. It is assumed that the DataFrame has a column named "filename" which contains the
    name of the video clip file in the format "original_movie_name_start_time.mp4". The function
    processes this information to extract
    :type meta_df: pd.DataFrame
    :return: a pandas DataFrame containing the filename of the clips, the filename of the original
    movie, the starting time of the clips in relation to the original movie, and the end time of the
    clips in relation to the original movie.
    """
    # Select the filename of the clips and remove extension type
    clip_filenames = meta_df["filename"].str.replace(".mp4", "", regex=True)

    # Get the starting time of clips in relation to the original movie
    # split the filename and select the last section
    meta_df["clip_start_time"] = clip_filenames.str.rsplit("_", 1).str[-1]

    # Extract the filename of the original movie
    meta_df["movie_filename"] = meta_df.apply(
        lambda x: x["filename"]
        .replace("_" + x["clip_start_time"], "")
        .replace(".mp4", ".mov"),
        axis=1,
    )

    # Get the end time of clips in relation to the original movie
    meta_df["clip_start_time"] = pd.to_numeric(
        meta_df["clip_start_time"], downcast="signed"
    )
    meta_df["clip_end_time"] = meta_df["clip_start_time"] + 10

    # Select only relevant columns
    meta_df = meta_df[
        ["filename", "movie_filename", "clip_start_time", "clip_end_time"]
    ]

    return meta_df


# Function to get the list of duplicated subjects
def get_duplicatesdf(project: Project):
    """
    This function reads a CSV file containing information about duplicated subjects and returns a pandas
    dataframe.
    
    :param project: The "project" parameter is an object of the "Project" class, which likely contains
    information about a specific project or dataset being worked on. The function is likely part of a
    larger program or script that uses the "project" object to access relevant information or data
    :type project: Project
    :return: a pandas DataFrame containing information about duplicated subjects in a project.
    """
    # Define the path to the csv files with initial info to build the db
    db_csv_info = project.csv_folder

    # Define the path to the csv file with ids of the duplicated subjects
    for file in Path(db_csv_info).rglob("*.csv"):
        if "duplicat" in file.name:
            duplicates_csv = file

    # Load the csv with information about duplicated subjects
    duplicatesdf = pd.read_csv(duplicates_csv)

    return duplicatesdf


# Function to select the first subject of those that are duplicated
def clean_duplicated_subjects(subjects: pd.DataFrame, project: Project):
    """
    This function takes a dataframe of subjects and a project, identifies and removes duplicated
    subjects, and returns a cleaned dataframe of unique subjects.
    
    :param subjects: A pandas DataFrame containing information about subjects in a project
    :type subjects: pd.DataFrame
    :param project: The `project` parameter is an instance of the `Project` class, which is not defined
    in the code snippet provided. It is likely that this class is defined elsewhere in the codebase and
    contains information about the project being worked on, such as project name, project ID, and
    project data
    :type project: Project
    :return: a cleaned dataframe of subjects with duplicated subjects removed and replaced with the id
    of the first subject.
    """
    # Get the duplicates df
    duplicatesdf = get_duplicatesdf(project)

    # Include a column with unique ids for duplicated subjects
    subjects = pd.merge(
        subjects,
        duplicatesdf,
        how="left",
        left_on="subject_id",
        right_on="dupl_subject_id",
    )

    # Replace the id of duplicated subjects for the id of the first subject
    subjects.subject_id = np.where(
        subjects.single_subject_id.isnull(),
        subjects.subject_id,
        subjects.single_subject_id,
    )

    # Select only unique subjects
    subjects = subjects.drop_duplicates(subset="subject_id", keep="first")

    return subjects


def process_koster_subjects(subjects: pd.DataFrame, db_path: str):
    """
    This function takes in a dataframe of subjects and a path to the database and returns a dataframe of
    subjects with updated metadata

    :param subjects: the dataframe of subjects from the database
    :type subjects: pd.DataFrame
    :param db_path: the path to the database
    :type db_path: str
    :return: A dataframe with all the subjects that have been uploaded to the database.
    """

    ## Set the date when the metadata of subjects uploaded matches/doesn't match schema.py requirements
    from kso_utils.db_utils import get_movies_id

    # Specify the date when the metadata of subjects uploaded matches schema.py
    auto_date = "2020-05-29 00:00:00 UTC"

    # Specify the starting date when clips were manually uploaded
    manual_date = "2019-11-17 00:00:00 UTC"

    ## Update subjects automatically uploaded

    # Select automatically uploaded subjects
    auto_subjects_df = auto_subjects(subjects, auto_date=auto_date)

    ## Update subjects manually uploaded
    # Select manually uploaded subjects
    manual_subjects_df = manual_subjects(
        subjects, manual_date=manual_date, auto_date=auto_date
    )

    # Include movie_ids to the metadata
    manual_subjects_df = get_movies_id(manual_subjects_df, db_path)

    # Combine all uploaded subjects
    subjects = pd.merge(manual_subjects_df, auto_subjects_df, how="outer")

    return subjects


# Function to combine classifications received on duplicated subjects
def combine_annot_from_duplicates(annot_df: pd.DataFrame, project: Project):
    """
    This function combines annotations from duplicate subjects in a DataFrame by replacing the IDs of
    duplicated subjects with the ID of the first subject.
    
    :param annot_df: a pandas DataFrame containing annotations for subjects in a project
    :type annot_df: pd.DataFrame
    :param project: Unfortunately, the parameter "project" is not defined in the code snippet provided.
    It is likely a custom object or variable specific to the project this code is being used for
    :type project: Project
    :return: a pandas DataFrame with the annotations from the input DataFrame `annot_df`, where the
    subject IDs of duplicated subjects have been replaced with the ID of the first subject. The function
    takes two arguments: `annot_df`, which is a pandas DataFrame with the annotations, and `project`,
    which is an instance of a Project class.
    """
    # Get the duplicates df
    duplicatesdf = get_duplicatesdf(project)

    # Include a column with unique ids for duplicated subjects
    annot_df = pd.merge(
        annot_df,
        duplicatesdf,
        how="left",
        left_on="subject_ids",
        right_on="dupl_subject_id",
    )

    # Replace the id of duplicated subjects for the id of the first subject
    annot_df["subject_ids"] = np.where(
        annot_df.single_subject_id.isnull(),
        annot_df.subject_ids,
        annot_df.single_subject_id,
    )

    return annot_df


def process_clips_koster(annotations, row_class_id: str, rows_list: list):
    """
    For each annotation, if the task is T4, then for each species annotated, flatten the relevant
    answers and save the species of choice, class and subject id

    :param annotations: the list of annotations for a given classification
    :param row_class_id: the classification id of the row
    :param rows_list: list
    :type rows_list: list
    :return: A list of dictionaries, each dictionary containing the classification id, the label, the first time seen and how many individuals were seen.
    """

    nothing_values = [
        "NOANIMALSPRESENT",
        "ICANTRECOGNISEANYTHING",
        "ISEENOTHING",
        "NOTHINGHERE",
    ]

    for ann_i in annotations:
        if ann_i["task"] == "T4":
            # Select each species annotated and flatten the relevant answers
            for value_i in ann_i["value"]:
                choice_i = {}
                # If choice = 'nothing here', set follow-up answers to blank
                if value_i["choice"] in nothing_values:
                    f_time = ""
                    inds = ""
                # If choice = species, flatten follow-up answers
                else:
                    answers = value_i["answers"]
                    f_time, inds = None, None
                    for k in answers.keys():
                        if "FIRSTTIME" in k:
                            f_time = answers[k].replace("S", "")
                        if "INDIVIDUAL" in k:
                            inds = answers[k]
                        elif "FIRSTTIME" not in k and "INDIVIDUAL" not in k:
                            f_time, inds = None, None

                # Save the species of choice, class and subject id
                choice_i.update(
                    {
                        "classification_id": row_class_id,
                        "label": value_i["choice"],
                        "first_seen": f_time,
                        "how_many": inds,
                    }
                )

                rows_list.append(choice_i)

    return rows_list


def process_koster_movies_csv(movies_df: pd.DataFrame):
    """
    It takes a dataframe of movies and returns a dataframe of movies with the following changes:

    - The filename is standardized
    - The filename is unswedified
    - The filename is renamed to fpath
    - The SamplingStart and SamplingEnd columns are renamed to sampling_start and sampling_end

    :param movies_df: the dataframe containing the movies
    :return: A dataframe with the columns:
        - filename
        - fpath
        - sampling_start
        - sampling_end
    """
    # Standarise the filename
    movies_df["filename"] = movies_df["filename"].str.normalize("NFD")

    # Unswedify the filename
    from kso_utils.movie_utils import unswedify

    movies_df["filename"] = movies_df["filename"].apply(lambda x: unswedify(x))

    # TO DO Include server's path to the movie files
    movies_df["fpath"] = movies_df["filename"]

    # Rename relevant fields
    movies_df = movies_df.rename(
        columns={
            "SamplingStart": "sampling_start",
            "SamplingEnd": "sampling_end",
        }
    )

    return movies_df


def bb_iou(boxA, boxB):
    """
    The function takes two bounding boxes, computes the area of intersection, and divides it by the area
    of the union of the two boxes

    :param boxA: The first bounding box
    :param boxB: The ground truth box
    :return: The IOU value
    """

    # Compute edges
    temp_boxA = boxA.copy()
    temp_boxB = boxB.copy()
    temp_boxA[2], temp_boxA[3] = (
        temp_boxA[0] + temp_boxA[2],
        temp_boxA[1] + temp_boxA[3],
    )
    temp_boxB[2], temp_boxB[3] = (
        temp_boxB[0] + temp_boxB[2],
        temp_boxB[1] + temp_boxB[3],
    )

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(temp_boxA[0], temp_boxB[0])
    yA = max(temp_boxA[1], temp_boxB[1])
    xB = min(temp_boxA[2], temp_boxB[2])
    yB = min(temp_boxA[3], temp_boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 1
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((temp_boxA[2] - temp_boxA[0]) * (temp_boxA[3] - temp_boxA[1]))
    boxBArea = abs((temp_boxB[2] - temp_boxB[0]) * (temp_boxB[3] - temp_boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return 1 - iou


def filter_bboxes(
    total_users: int, users: list, bboxes: list, obj: float, eps: float, iua: float
):
    """
    If at least half of the users who saw this frame decided that there was an object, then we cluster
    the bounding boxes based on the IoU criterion. If at least 80% of users agree on the annotation,
    then we accept the cluster assignment

    :param total_users: total number of users who saw this frame
    :param users: list of user ids
    :param bboxes: list of bounding boxes
    :param obj: the minimum fraction of users who must have seen an object in order for it to be considered
    :param eps: The maximum distance between two samples for them to be considered as in the same neighborhood
    :param iua: the minimum percentage of users who must agree on a bounding box for it to be accepted
    """

    # If at least half of those who saw this frame decided that there was an object
    user_count = pd.Series(users).nunique()
    if user_count / total_users >= obj:
        # Get clusters of annotation boxes based on iou criterion
        cluster_ids = DBSCAN(min_samples=1, metric=bb_iou, eps=eps).fit_predict(bboxes)
        # Count the number of users within each cluster
        counter_dict = Counter(cluster_ids)
        # Accept a cluster assignment if at least 80% of users agree on annotation
        passing_ids = [k for k, v in counter_dict.items() if v / user_count >= iua]

        indices = np.isin(cluster_ids, passing_ids)

        final_boxes = []
        for i in passing_ids:
            # Compute median over all accepted bounding boxes
            boxes = np.median(np.array(bboxes)[np.where(cluster_ids == i)], axis=0)
            final_boxes.append(boxes)

        return indices, final_boxes

    else:
        return [], bboxes
