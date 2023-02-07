# base imports
import os
import cv2
import pandas as pd
import logging
import imghdr
from tqdm import tqdm
from pathlib import Path
import splitfolders

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def create_classification_dataset(data_path: str, out_path: str, test_size: float):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    splitfolders.ratio(data_path, output=out_path, seed=1337, ratio=(1-test_size, 0, test_size))
    logging.info(f"Training and test datasets saved at {out_path}") 


def get_patches(root_path: str, meta_filename: str, pixels: int, out_path: str):
    """
    The function takes as input a folder with images, a metadata-sheet, a height/width in pixels, and an
    output path, and gives as output square patches from all points specified in the sheet, with size
    equal to pixels:pixels

    :param root_path: the path to the folder containing the images and the metadata-sheet
    :type root_path: str
    :param meta_filename: the name of the metadata file, which is an excel file
    :type meta_filename: str
    :param pixels: the size of the square patch you want to extract (usually 224)
    :type pixels: int
    :param out_path: the path to the folder where you want to save the patches
    :type out_path: str
    :return: nothing, but it creates a folder with patches from the images in the root_path folder.
    """

    path_to_folder = Path(root_path)
    # get list of image files
    image_list = [
        f
        for f in os.listdir(path_to_folder)
        if imghdr.what(Path(path_to_folder, f)) is not None
    ]

    # df: dataframe based on SGU metadata-sheet
    df = pd.read_excel(Path(path_to_folder, meta_filename))

    # create patch folder
    if not os.path.exists(f"{out_path}"):
        os.mkdir(f"{out_path}")


    k = 0
    for row in tqdm(df.itertuples()):
        if row.image_name in image_list:

            # Use conversion between current XY position and actual pixel values
            coord = (row.pos_X / 15, row.pos_Y / 15)

            # Discard images where pos_X is negative
            if coord[0] < 0:
                logging.error(f"Negative X value in {row.image_name}. Skipping...")
                pass

            # Load image
            img = cv2.imread(Path(path_to_folder, row.image_name))

            # Specify cropped patch size
            cropped_image = img[
                int(coord[0] - pixels / 2) : int(coord[0] + pixels / 2),
                int(coord[1] - pixels / 2) : int(coord[1] + pixels / 2),
            ]

            # Get label
            label = row.sub_type
            if not os.path.exists(Path(out_path, label)):
                os.mkdir(Path(out_path, label))

            # Write patches to a folder
            k += 1
            cv2.imwrite(
                f"{Path(out_path, label)}/{Path(data[0]).stem}_patch_{k}.jpg", cropped_image
            )

    logging.info(
        f"Patch creation completed successfully. Total patches: {len(Path(out_path).iterdir())}"
    )


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
